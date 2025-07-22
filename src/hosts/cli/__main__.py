"""A2A Host CLI using Typer."""

import asyncio
import base64
import os
import urllib
from typing import Annotated
from uuid import uuid4

import httpx
import typer
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    FilePart,
    FileWithBytes,
    GetTaskRequest,
    JSONRPCErrorResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskQueryParams,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

from common.utils.push_notification_auth import PushNotificationReceiverAuth
from hosts.cli.push_notification_listener import PushNotificationListener

app = typer.Typer()


async def complete_task(
    client: A2AClient,
    streaming: bool,
    use_push_notifications: bool,
    notification_receiver_host: str,
    notification_receiver_port: int,
    task_id: str | None,
    context_id: str,
) -> tuple[bool, str | None, str | None]:
    """Complete a single task interaction.

    Args:
        client: A2A client instance
        streaming: Whether to use streaming
        use_push_notifications: Whether to use push notifications
        notification_receiver_host: Host for push notifications
        notification_receiver_port: Port for push notifications
        task_id: Optional task ID
        context_id: Context ID

    Returns:
        Tuple of (continue_loop, context_id, task_id)
    """
    prompt = typer.prompt(
        "\nWhat do you want to send to the agent? (:q or quit to exit)"
    )
    if prompt == ":q" or prompt == "quit":
        return False, None, None

    message = Message(
        role="user",
        parts=[TextPart(text=prompt)],
        messageId=str(uuid4()),
        taskId=task_id,
        contextId=context_id,
    )

    file_path = typer.prompt(
        "Select a file path to attach? (press enter to skip)",
        default="",
        show_default=False,
    )
    if file_path and file_path.strip() != "":
        with open(file_path, "rb") as f:
            file_content = base64.b64encode(f.read()).decode("utf-8")
            file_name = os.path.basename(file_path)

        message.parts.append(
            Part(root=FilePart(file=FileWithBytes(name=file_name, bytes=file_content)))
        )

    payload = MessageSendParams(
        id=str(uuid4()),
        message=message,
        configuration=MessageSendConfiguration(
            acceptedOutputModes=["text"],
        ),
    )

    if use_push_notifications:
        payload["pushNotification"] = {
            "url": f"http://{notification_receiver_host}:{notification_receiver_port}/notify",
            "authentication": {
                "schemes": ["bearer"],
            },
        }

    task_result = None
    message = None
    task_completed = False

    if streaming:
        response_stream = client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=payload,
            )
        )
        async for result in response_stream:
            if isinstance(result.root, JSONRPCErrorResponse):
                print(
                    f"Error: {result.root.error}, contextId: {context_id}, taskId: {task_id}"
                )
                return False, context_id, task_id
            event = result.root.result
            context_id = event.contextId
            if isinstance(event, Task):
                task_id = event.id
            elif isinstance(event, TaskStatusUpdateEvent) or isinstance(
                event, TaskArtifactUpdateEvent
            ):
                task_id = event.taskId
                if (
                    isinstance(event, TaskStatusUpdateEvent)
                    and event.status.state == "completed"
                ):
                    task_completed = True
            elif isinstance(event, Message):
                message = event
            print(f"stream event => {event.model_dump_json(exclude_none=True)}")

        # Upon completion of the stream, retrieve the full task if one was made
        if task_id and not task_completed:
            task_result_response = await client.get_task(
                GetTaskRequest(
                    id=str(uuid4()),
                    params=TaskQueryParams(id=task_id),
                )
            )
            if isinstance(task_result_response.root, JSONRPCErrorResponse):
                print(
                    f"Error: {task_result_response.root.error}, contextId: {context_id}, taskId: {task_id}"
                )
                return False, context_id, task_id
            task_result = task_result_response.root.result
    else:
        try:
            # For non-streaming, assume the response is a task or message
            event = await client.send_message(
                SendMessageRequest(
                    id=str(uuid4()),
                    params=payload,
                )
            )
            event = event.root.result
        except Exception as e:
            print("Failed to complete the call", e)
            return True, context_id, task_id

        if not context_id:
            context_id = event.contextId
        if isinstance(event, Task):
            if not task_id:
                task_id = event.id
            task_result = event
        elif isinstance(event, Message):
            message = event

    if message:
        print(f"\n{message.model_dump_json(exclude_none=True)}")
        return True, context_id, task_id

    if task_result:
        # Don't print the contents of a file
        task_content = task_result.model_dump_json(
            exclude={
                "history": {
                    "__all__": {
                        "parts": {
                            "__all__": {"file"},
                        },
                    },
                },
            },
            exclude_none=True,
        )
        print(f"\n{task_content}")
        # If the result is that more input is required, loop again
        state = TaskState(task_result.status.state)
        if state.name == TaskState.input_required.name:
            return (
                await complete_task(
                    client,
                    streaming,
                    use_push_notifications,
                    notification_receiver_host,
                    notification_receiver_port,
                    task_id,
                    context_id,
                ),
                context_id,
                task_id,
            )
        # Task is complete
        return True, context_id, task_id

    # Failure case, shouldn't reach
    return True, context_id, task_id


async def async_main(
    agent: str = "http://localhost:10000",
    bearer_token: str | None = None,
    session: int = 0,
    history: bool = False,
    use_push_notifications: bool = False,
    push_notification_receiver: str = "http://localhost:5000",
    headers: list[str] = None,
) -> None:
    """Main async function for the A2A host CLI.

    Args:
        agent: Agent URL
        bearer_token: Bearer token for authentication
        session: Session ID
        history: Whether to show history
        use_push_notifications: Whether to use push notifications
        push_notification_receiver: Push notification receiver URL
        headers: Additional headers
    """
    # Parse headers
    request_headers = {}
    if headers:
        for h in headers:
            key, value = h.split("=", 1)
            request_headers[key] = value

    if bearer_token:
        request_headers["Authorization"] = f"Bearer {bearer_token}"

    print(f"Will use headers: {request_headers}")

    async with httpx.AsyncClient(timeout=30, headers=request_headers) as httpx_client:
        card_resolver = A2ACardResolver(httpx_client, agent)
        card = await card_resolver.get_agent_card()

        print("======= Agent Card ========")
        print(card.model_dump_json(exclude_none=True))

        notif_receiver_parsed = urllib.parse.urlparse(push_notification_receiver)
        notification_receiver_host = notif_receiver_parsed.hostname
        notification_receiver_port = notif_receiver_parsed.port

        if use_push_notifications:
            notification_receiver_auth = PushNotificationReceiverAuth()
            await notification_receiver_auth.load_jwks(f"{agent}/.well-known/jwks.json")

            push_notification_listener = PushNotificationListener(
                host=notification_receiver_host,
                port=notification_receiver_port,
                notification_receiver_auth=notification_receiver_auth,
            )
            push_notification_listener.start()

        client = A2AClient(httpx_client, agent_card=card)

        continue_loop = True
        streaming = card.capabilities.streaming
        context_id = session if session > 0 else uuid4().hex

        while continue_loop:
            print("=========  starting a new task ======== ")
            continue_loop, _, task_id = await complete_task(
                client,
                streaming,
                use_push_notifications,
                notification_receiver_host,
                notification_receiver_port,
                None,
                context_id,
            )

            if history and continue_loop:
                print("========= history ======== ")
                task_response = await client.get_task(
                    {"id": task_id, "historyLength": 10}
                )
                print(
                    task_response.model_dump_json(include={"result": {"history": True}})
                )


@app.command()
def main(
    agent: Annotated[str, typer.Option(help="Agent URL")] = "http://localhost:10000",
    bearer_token: Annotated[
        str | None,
        typer.Option(
            help="Bearer token for authentication",
            envvar="A2A_CLI_BEARER_TOKEN",
        ),
    ] = None,
    session: Annotated[int, typer.Option(help="Session ID")] = 0,
    history: Annotated[bool, typer.Option(help="Show task history")] = False,
    use_push_notifications: Annotated[
        bool, typer.Option(help="Enable push notifications")
    ] = False,
    push_notification_receiver: Annotated[
        str, typer.Option(help="Push notification receiver URL")
    ] = "http://localhost:5000",
    header: Annotated[
        list[str] | None, typer.Option(help="Additional headers (format: key=value)")
    ] = None,
) -> None:
    """A2A Host CLI - interact with A2A agents."""
    asyncio.run(
        async_main(
            agent,
            bearer_token,
            session,
            history,
            use_push_notifications,
            push_notification_receiver,
            header or [],
        )
    )


if __name__ == "__main__":
    app()
