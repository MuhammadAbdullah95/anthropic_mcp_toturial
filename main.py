import asyncio
import sys
import os
import warnings
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from typing import Optional

# Suppress Windows-specific asyncio warnings
if sys.platform == "win32":
    # Set environment variable to suppress these specific warnings
    os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning"
    # Also filter through warnings module
    warnings.filterwarnings("ignore", category=ResourceWarning)

from mcp_client import MCPClient
# Replace Anthropic/Claude with Google Gemini
from google import genai

from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

# Gemini Config
gemini_model = os.getenv("GEMINI_MODEL", "")
gemini_api_key = os.getenv("GEMINI_API_KEY", "")


assert gemini_model, "Error: GEMINI_MODEL cannot be empty. Update .env"
assert gemini_api_key, "Error: GEMINI_API_KEY cannot be empty. Update .env"

# Initialize Gemini client
gemini_client = genai.Client(api_key=gemini_api_key)


class MockResponse:
    """Mock response object to match expected interface."""
    def __init__(self, text="", function_calls=None):
        self.text = text
        self.function_calls = function_calls or []


class GeminiService:
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model

    async def generate(self, prompt: str) -> str:
        # Use the new Google GenAI SDK API
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=512
            )
        )
        return str(response.text)

    async def chat_with_tools(self, messages: list, tools: Optional[list] = None):
        """Generate content with tool support for function calling."""
        # Convert our message format to Gemini's format
        gemini_contents = self._convert_messages_to_gemini_format(messages)
        
        # Build config with tools if provided
        gemini_tools = None
        if tools:
            # Convert tools to Gemini format
            gemini_tools = [genai.types.Tool(function_declarations=tools)]
        
        config = genai.types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1024,
            tools=gemini_tools  # type: ignore
        )
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )
            
            # Extract function calls if any
            function_calls = []
            response_text = ""
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            # Convert to our expected format
                            func_call = MockFunctionCall(
                                name=str(part.function_call.name),
                                args=dict(part.function_call.args) if part.function_call.args else {}
                            )
                            function_calls.append(func_call)
                        elif hasattr(part, 'text') and part.text:
                            response_text += part.text
            
            # If no text and no function calls, try to get text from response object
            if not response_text and not function_calls:
                response_text = str(response.text) if hasattr(response, 'text') else ""
            
            return MockResponse(text=response_text, function_calls=function_calls)
            
        except Exception as e:
            print(f"Error in chat_with_tools: {e}")
            # Fallback to simple text generation
            fallback_response = await self.generate(self._convert_messages_to_simple_prompt(messages))
            return MockResponse(text=fallback_response)

    def _convert_messages_to_gemini_format(self, messages: list) -> list:
        """Convert our message format to Gemini's expected format."""
        gemini_contents = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Convert role: user/assistant -> user/model
            gemini_role = "model" if role == "assistant" else "user"
            
            if isinstance(content, str):
                # Simple text content
                gemini_contents.append(
                    genai.types.Content(
                        role=gemini_role,
                        parts=[genai.types.Part(text=content)]
                    )
                )
            elif isinstance(content, list):
                # Handle complex content with multiple parts
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(genai.types.Part(text=item.get("text", "")))
                        elif item.get("type") == "function_call":
                            # Handle function call parts
                            func_call = item.get("function_call", {})
                            # Note: Gemini may handle this differently, but we'll include as text for now
                            parts.append(genai.types.Part(text=f"Function call: {func_call.get('name', 'unknown')}"))
                    else:
                        # Fallback: convert to string
                        parts.append(genai.types.Part(text=str(item)))
                
                if parts:
                    gemini_contents.append(
                        genai.types.Content(role=gemini_role, parts=parts)
                    )
        
        return gemini_contents

    def _convert_messages_to_simple_prompt(self, messages: list) -> str:
        """Convert messages to a simple prompt string for fallback."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                prompt_parts.append(f"{role}: {content}")
            else:
                prompt_parts.append(f"{role}: {str(content)}")
        return "\n".join(prompt_parts)


class MockFunctionCall:
    """Mock function call object to match expected interface."""
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args


async def main():
    gemini_service = GeminiService(
        client=gemini_client,
        model=gemini_model
    )

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            ai_service=gemini_service  # Pass Gemini service as ai_service
        )

        cli = CliApp(chat)
        await cli.initialize()
        try:
            await cli.run()
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        finally:
            # Give subprocess transports time to clean up on Windows
            if sys.platform == "win32":
                await asyncio.sleep(0.1)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
