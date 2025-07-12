from typing import Protocol, Union, Optional
from mcp_client import MCPClient
from core.tools import ToolManager
from anthropic.types import MessageParam


class AIService(Protocol):
    """Protocol for AI services that can be used with Chat."""
    async def generate(self, prompt: str) -> str:
        ...
        
    async def chat_with_tools(self, messages: list, tools: Optional[list] = None):
        """Generate content with tool support. Should return object with function_calls attribute and text."""
        ...


class Chat:
    def __init__(self, ai_service: AIService, clients: dict[str, MCPClient]):
        self.ai_service: AIService = ai_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[MessageParam] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        final_text_response = ""

        await self._process_query(query)

        max_iterations = 5  # Prevent infinite loops
        iteration_count = 0
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            # Get all available tools from MCP clients
            tools = await ToolManager.get_all_tools(self.clients)
            
            # Generate response with tools (using Gemini's function calling)
            response = await self.ai_service.chat_with_tools(
                messages=self.messages,
                tools=tools,
            )

            # Add the assistant's response to conversation
            self._add_assistant_message(response)

            # Check if the model wants to call functions
            if hasattr(response, 'function_calls') and response.function_calls:
                print(f"AI wants to call tools: {[fc.name for fc in response.function_calls]}")
                
                # Execute the tool requests
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )
                
                print(f"Tool execution completed. Results: {len(tool_result_parts)} items")
                for i, result in enumerate(tool_result_parts):
                    print(f"  Result {i+1}: {result.get('tool_use_id', 'unknown')} - {'Error' if result.get('is_error') else 'Success'}")

                # Add tool results to conversation  
                self._add_user_message(tool_result_parts)
            else:
                # No more function calls, we have the final response
                final_text_response = response.text if hasattr(response, 'text') else str(response)
                break

        if iteration_count >= max_iterations:
            print(f"Warning: Stopped after {max_iterations} iterations to prevent infinite loop")
            final_text_response = "I encountered an issue while processing your request. Please try rephrasing your question."

        return final_text_response

    def _add_assistant_message(self, response):
        """Add assistant message to conversation history."""
        if hasattr(response, 'function_calls') and response.function_calls:
            # If there are function calls, add them to the message
            # This is similar to how Claude's implementation works
            content = []
            if hasattr(response, 'text') and response.text:
                content.append({"type": "text", "text": response.text})
            
            # Add function calls (Gemini format)
            for func_call in response.function_calls:
                content.append({
                    "type": "function_call", 
                    "function_call": {
                        "name": func_call.name,
                        "args": func_call.args
                    }
                })
            
            self.messages.append({"role": "assistant", "content": content})
        else:
            # Regular text response
            text_content = response.text if hasattr(response, 'text') else str(response)
            self.messages.append({"role": "assistant", "content": text_content})

    def _add_user_message(self, tool_result_parts):
        """Add tool execution results to conversation history."""
        print(f"Adding tool results to conversation: {tool_result_parts}")
        
        # Convert tool results to a format Gemini can understand
        if tool_result_parts:
            # Create a simple text summary of the tool results
            result_text = "Tool execution results:\n"
            for result in tool_result_parts:
                tool_id = result.get('tool_use_id', 'unknown')
                content = result.get('content', 'No content')
                is_error = result.get('is_error', False)
                status = "ERROR" if is_error else "SUCCESS"
                result_text += f"- {tool_id}: {status} - {content}\n"
            
            self.messages.append({"role": "user", "content": result_text})
        else:
            self.messages.append({"role": "user", "content": "Tool execution completed with no results."})
