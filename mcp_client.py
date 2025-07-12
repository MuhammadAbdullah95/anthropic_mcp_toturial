import sys
import json
from pydantic import AnyUrl
import asyncio
import warnings
from typing import Optional, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# Suppress Windows-specific asyncio warnings
if sys.platform == "win32":
    import os
    # Set environment variable to suppress these specific warnings
    os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning"
    # Also filter through warnings module
    warnings.filterwarnings("ignore", category=ResourceWarning)


class MCPClient:
    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[dict] = None,
    ):
        self._command = command
        self._args = args
        self._env = env
        self._session: Optional[ClientSession] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self):
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        _stdio, _write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(_stdio, _write)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError(
                "Client session not initialized or cache not populated. Call connect_to_server first."
            )
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        # TODO: Return a list of tools defined by the MCP server
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(
        self, tool_name: str, tool_input: dict
    ) -> types.CallToolResult | None:
        # TODO: Call a particular tool and return the result
        return await self.session().call_tool(tool_name, tool_input)

    async def list_prompts(self) -> list[types.Prompt]:
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name, args: dict[str, str]):
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages

    async def read_resource(self, uri: str) -> Any:
        result = await self.session().read_resource(AnyUrl(uri))
        response = result.contents[0]

        if isinstance(response, types.TextResourceContents):
            if response.mimeType == "application/json":
                return json.loads(response.text)
            
            return response.text
    async def cleanup(self):
        try:
            if self._session:
                # Give the session a moment to clean up properly
                await asyncio.sleep(0.1)
            await self._exit_stack.aclose()
            # Give subprocess transports time to clean up on Windows
            if sys.platform == "win32":
                await asyncio.sleep(0.1)
        except Exception as e:
            # Ignore cleanup exceptions to prevent noise
            pass
        finally:
            self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


# For testing
async def main():
    async with MCPClient(
        # If using Python without UV, update command to 'python' and remove "run" from args.
        command="uv",
        args=["run", "mcp_server.py"],
    ) as client:
        print("Listing tools:")
        tools = await client.list_tools()
        print(tools)
        print("-" * 50)
        print("Calling tool:")
        result = await client.call_tool("read_doc_contents", {"doc_id": "deposition.md"})
        print(result)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
