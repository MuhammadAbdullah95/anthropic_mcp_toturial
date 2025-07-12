import json
from typing import Optional, Literal, List, Dict, Any
from mcp.types import CallToolResult, Tool, TextContent
from mcp_client import MCPClient
from anthropic.types import Message, ToolResultBlockParam


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict]:
        """Gets all tools from the provided clients in Gemini function declaration format."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            for t in tool_models:
                # Convert MCP tool to Gemini function declaration format
                gemini_tool: Dict[str, Any] = {
                    "name": t.name,
                    "description": t.description,
                }
                
                # Only add parameters if they exist and convert to Gemini format
                if hasattr(t, 'inputSchema') and t.inputSchema:
                    # Clean the input schema to match Gemini's expected format
                    parameters = cls._convert_mcp_schema_to_gemini(t.inputSchema)
                    if parameters and isinstance(parameters, dict):
                        gemini_tool["parameters"] = parameters
                
                tools.append(gemini_tool)
        return tools
    
    @classmethod
    def _convert_mcp_schema_to_gemini(cls, mcp_schema: dict) -> dict:
        """Convert MCP input schema to Gemini function parameters format."""
        if not isinstance(mcp_schema, dict):
            return {}
        
        # Start with basic structure
        gemini_schema = {
            "type": mcp_schema.get("type", "object")
        }
        
        # Add properties if they exist
        if "properties" in mcp_schema:
            gemini_schema["properties"] = {}
            for prop_name, prop_def in mcp_schema["properties"].items():
                if isinstance(prop_def, dict):
                    # Clean property definition - only keep fields Gemini supports
                    clean_prop = {
                        "type": prop_def.get("type", "string")
                    }
                    if "description" in prop_def:
                        clean_prop["description"] = prop_def["description"]
                    if "enum" in prop_def:
                        clean_prop["enum"] = prop_def["enum"]
                    
                    gemini_schema["properties"][prop_name] = clean_prop
        
        # Add required fields if they exist
        if "required" in mcp_schema:
            gemini_schema["required"] = mcp_schema["required"]
        
        return gemini_schema

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that has the specified tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    def _build_tool_result_part(
        cls,
        tool_use_id: str,
        text: str,
        status: Literal["success"] | Literal["error"],
    ) -> ToolResultBlockParam:
        """Builds a tool result part dictionary."""
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": text,
            "is_error": status == "error",
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], response
    ) -> List[ToolResultBlockParam]:
        """
        Executes tool requests from either Claude-style (Message) or Gemini-style (response with function_calls).
        """
        # Check if this is a Gemini-style response with function_calls
        if hasattr(response, 'function_calls') and response.function_calls:
            return await cls._execute_gemini_function_calls(clients, response.function_calls)
        
        # Fall back to Claude-style tool_use blocks
        if hasattr(response, 'content'):
            tool_requests = [
                block for block in response.content if hasattr(block, 'type') and block.type == "tool_use"
            ]
            return await cls._execute_claude_tool_requests(clients, tool_requests)
        
        # No tools to execute
        return []

    @classmethod
    async def _execute_gemini_function_calls(
        cls, clients: dict[str, MCPClient], function_calls: list
    ) -> List[ToolResultBlockParam]:
        """Execute Gemini-style function calls."""
        tool_result_blocks: list[ToolResultBlockParam] = []
        
        for i, func_call in enumerate(function_calls):
            tool_name = func_call.name
            tool_input = func_call.args
            tool_use_id = f"gemini_call_{i}_{tool_name}"  # Generate ID for Gemini calls

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id, "Could not find that tool", "error"
                )
                tool_result_blocks.append(tool_result_part)
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                items = []
                if tool_output:
                    items = tool_output.content
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_json = json.dumps(content_list)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    content_json,
                    "error"
                    if tool_output and tool_output.isError
                    else "success",
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    json.dumps({"error": error_message}),
                    "error",
                )

            tool_result_blocks.append(tool_result_part)
        
        return tool_result_blocks

    @classmethod
    async def _execute_claude_tool_requests(
        cls, clients: dict[str, MCPClient], tool_requests: list
    ) -> List[ToolResultBlockParam]:
        """Execute Claude-style tool requests."""
        tool_result_blocks: list[ToolResultBlockParam] = []
        
        for tool_request in tool_requests:
            tool_use_id = tool_request.id
            tool_name = tool_request.name
            tool_input = tool_request.input

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id, "Could not find that tool", "error"
                )
                tool_result_blocks.append(tool_result_part)
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                items = []
                if tool_output:
                    items = tool_output.content
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_json = json.dumps(content_list)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    content_json,
                    "error"
                    if tool_output and tool_output.isError
                    else "success",
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    json.dumps({"error": error_message}),
                    "error",
                )

            tool_result_blocks.append(tool_result_part)
        
        return tool_result_blocks
