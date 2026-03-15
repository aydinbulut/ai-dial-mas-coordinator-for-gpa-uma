import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        #TODO:
        # 1. Create AsyncDial client (api_version='2025-01-01-preview')
        async_dial_client = AsyncDial(
            base_url=self.endpoint, 
            api_key=request.api_key, 
            api_version='2025-01-01-preview')
        # 2. Open stage for Coordination Request (StageProcessor will help with that)
        stage_processor = StageProcessor()
        stage = stage_processor.open_stage(choice=choice, name="coordination_request")
        # 3. Prepare coordination request
        coordination_request = await self.__prepare_coordination_request(client=async_dial_client, request=request)
        # 4. Add to the stage generated coordination request and close the stage
        stage.append_content(json.dumps(coordination_request.model_dump(), indent=4))
        # 5. Handle coordination request (don't forget that all the content that will write called agent need to provide to stage)
        coordination_request_response = await self.__handle_coordination_request(
            coordination_request=coordination_request,
            choice=choice,
            stage=stage,
            request=request
        )
        stage_processor.close_stage_safely(stage=stage)
        # 6. Generate final response based on the message from called agent
        return await self.__final_response(
            client=async_dial_client,
            choice=choice,
            request=request,
            agent_message=coordination_request_response
        )

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        #TODO:
        # 1. Make call to LLM with prepared messages and COORDINATION_REQUEST_SYSTEM_PROMPT. For GPT model we can use
        #    `response_format` https://platform.openai.com/docs/guides/structured-outputs?example=structured-data and
        #    response will be returned in JSON format. The `response_format` parameter must be provided as extra_body dict
        #    {response_format": {"type": "json_schema","json_schema": {"name": "response","schema": CoordinationRequest.model_json_schema()}}}
        response = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=self.__prepare_messages(request=request, system_prompt=COORDINATION_REQUEST_SYSTEM_PROMPT),
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                }
            }
        )
        # 2. Get content from response -> choice -> message -> content
        content = response.choices[0].message.content
        # 3. Load as dict
        content_dict = json.loads(content)
        # 4. Create CoordinationRequest from result, since CoordinationRequest is pydentic model, you can use `model_validate` method
        return CoordinationRequest.model_validate(content_dict)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        #TODO:
        # 1. Create array with messages, first message is system prompt and it is dict
        messages = [{"role": Role.SYSTEM, "content": system_prompt}]
        # 2. Iterate through messages from request and:
        #       - if user message that it has custom content and then add dict with user message and content (custom_content should be skipped)
        #       - otherwise append it as dict with excluded none fields (use `dict` method, despite it is deprecated since
        #         DIAL is using pydentic.v1)
        for message in request.messages:
            if message.role == Role.USER and message.custom_content is not None:
                messages.append({
                    "role": Role.USER.value,
                    "content": message.content
                })
            else:
                messages.append(message.model_dump(exclude_none=True))

        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        #TODO:
        # Make appropriate coordination requests to to proper agents and return the result
        if coordination_request.agent_name == AgentName.GPA:
            gpa_gateway = GPAGateway(endpoint=self.endpoint)
            return await gpa_gateway.response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions
            )
        elif coordination_request.agent_name == AgentName.UMS:
            ums_agent_gateway = UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint)
            return await ums_agent_gateway.response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions
            )
        else:
            raise ValueError(f"Unknown agent name: {coordination_request.agent_name}")

    async def __final_response(
            self, client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        #TODO:
        # 1. Prepare messages with FINAL_RESPONSE_SYSTEM_PROMPT
        messages = self.__prepare_messages(request=request, system_prompt=FINAL_RESPONSE_SYSTEM_PROMPT)
        # 2. Make augmentation of retrieved agent response (as context) with user request (as user request)
        updated_user_request = f"## CONTEXT:\n {agent_message.content}\n ---\n ## USER_REQUEST: \n {messages[-1]["content"]}"
        # 3. Update last message content with augmented prompt
        messages[-1]["content"] = updated_user_request
        # 4. Call LLM with streaming
        chunks = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            stream=True
        )

        # 5. Stream final response to choice
        content = ""
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content
        
        return Message(
            role=Role.ASSISTANT,
            content=content,
            custom_content=agent_message.custom_content
        )
