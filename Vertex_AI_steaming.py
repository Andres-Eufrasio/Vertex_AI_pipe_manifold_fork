"""
title: Multi-Model Manifold Pipe with Safety Settings with gemini
author: Andres Eufrasio (andres.eufrasiotinajero@gmail.com)
version: 0.0.3
license: MIT

Decsiption:
Fork based on Feynman Liang (feynman@blueteam.ai
this is a Manifold Pipe for google vertex that has added functionality so that global servers work
control the model's values based on the constants at the top.
"""

import json
from pydantic import BaseModel, Field
from typing import Union, Iterator, List
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

# Set DEBUG to True to enable detailed logging
DEBUG = False

# Set these to easily control your model
TEMP = 1.7
TOPP = 0.98
TOPK = 64


class Pipe:
    class Valves(BaseModel):
        PROJECT_ID: str = Field(
            default="",  # Set this to your Project ID
            description="Your Google Cloud Project ID.",
        )
        LOCATION: str = Field(
            default="", description="Default region for Vertex AI models."
        )
        SERVICE_ACCOUNT_JSON: str = Field(
            default="", description="Service Account JSON"
        )

    def __init__(self):
        self.id = "vertex_ai_streaming"  # Set this to your function's ID in OpenWebUI
        self.type = "manifold"
        self.name = "Vertex "
        self.valves = self.Valves()
        self.credentials = None  # Initialize credentials attribute
        self.authed_session = None

    def _ensure_auth(self):
        # Only authenticate if session is not already created or credentials changed
        if not self.authed_session or not self.credentials:
            if not self.valves.SERVICE_ACCOUNT_JSON:
                raise ValueError("SERVICE_ACCOUNT_JSON is not configured in Valves.")
            try:
                credentials_info = json.loads(self.valves.SERVICE_ACCOUNT_JSON)
                self.credentials = (
                    service_account.Credentials.from_service_account_info(
                        credentials_info,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                )
                self.authed_session = AuthorizedSession(self.credentials)
                if DEBUG:
                    print("Authentication successful.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding SERVICE_ACCOUNT_JSON: {e}")
            except Exception as e:
                raise ConnectionError(f"Failed to authenticate with Google Cloud: {e}")

    def pipes(self) -> List[dict]:
        # Return a list of models
        return [
            {"id": "gemini-2.5-pro-preview-06-05", "name": "Gemini-2.5-pro-preview"},
            {"id": "gemini-3.1-pro-preview", "name": "Gemini-3.1-prev"},
            # Eample commented-out models
            # {"id": "claude-3-5-sonnet-v2@20241022", "name": "Claude 3.5 Sonnet V2"},
            # {"id": "claude-3-haiku@20240307", "name": "Claude 3 Haiku"},
            # {"id": "llama3-8b-instruct-tune", "name": "Llama 3 8B Instruct"}, # Example Llama model
        ]

    def pipe(self, body: dict) -> Union[str, Iterator[str]]:
        try:
            self._ensure_auth()  # Ensure authentication is done first
        except (ValueError, ConnectionError) as e:
            if DEBUG:
                print(f"Authentication error: {e}")
            return f"Error: {e}"

        project_id = self.valves.PROJECT_ID
        default_location = self.valves.LOCATION

        if not project_id:
            return "Error: PROJECT_ID is not configured in Valves."
        if not default_location:
            return "Error: LOCATION is not configured in Valves."

        model_id = body.get("model")
        stream = body.get("stream", False)

        if not model_id:
            return "Error: No model specified."

        if DEBUG:
            print(f"Original model_id: {model_id}")

        # Remove the function ID prefix if present
        function_id_with_dot = f"{self.id}."
        if model_id.startswith(function_id_with_dot):
            model_id = model_id[len(function_id_with_dot) :]
            if DEBUG:
                print(
                    f"Removed prefix '{function_id_with_dot}' from model_id, new model_id: {model_id}"
                )
        else:
            if DEBUG:
                print("No prefix to remove from model_id.")

        if DEBUG:
            print(f"Using model_id: {model_id}")

        try:
            # Determine which model is selected and call the appropriate method
            if model_id.startswith("gemini"):
                return self.call_gemini_model(
                    body, project_id, default_location, model_id
                )
            elif model_id.startswith("claude"):
                # Use 'us-east5' for Claude models - Check if this is still the recommended region
                claude_location = "us-east5"
                if DEBUG:
                    print(
                        f"Using location '{claude_location}' for Claude model: {model_id}"
                    )
                return self.call_claude_model(
                    body, project_id, claude_location, model_id
                )
            elif model_id.startswith("llama"):
                # Llama models might use the default location or a specific one
                # Assuming default location here, adjust if needed
                return self.call_llama_model(
                    body, project_id, default_location, model_id
                )
            else:
                return f"Error: Unknown or unsupported model ID: {model_id}"
        except Exception as e:
            if DEBUG:
                import traceback

                print(f"Error in pipe method: {e}\n{traceback.format_exc()}")
            return f"Error: An unexpected error occurred processing the request for model {model_id}. Details: {e}"

    def call_gemini_model(self, body, project_id, location, model_id):
        # Use the OpenAPI-spec compliant endpoint for chat completions
        if location == "global":
            url = (
                f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:streamGenerateContent?alt=sse"
                if body.get("stream", False)
                else f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"
            )
        else:
            url = (
                f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:streamGenerateContent?alt=sse"
                if body.get("stream", False)
                else f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"
            )

        # --- Safety Settings Definition ---
        # Define safety settings using the JSON structure expected by the REST API.
        # Assuming 'BLOCK_NONE' as the threshold blocks nothing for that category.
        # Adjust thresholds (e.g., BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH)
        # based on your requirements.
        safety_settings_config = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
        ]
        # --- End Safety Settings Definition ---

        # Prepare the messages in the format required by the generateContent API
        messages = body["messages"]
        formatted_contents = self.format_gemini_contents(
            messages
        )  # Use specific Gemini formatting

        # Prepare request data for the generateContent API
        data = {
            "contents": formatted_contents,
            "generationConfig": {
                "temperature": body.get(
                    "temperature", TEMP
                ),  # Default T for Gemini 1.5 Pro often higher
                "maxOutputTokens": body.get(
                    "max_tokens", 4240
                ),  # Gemini 1.5 Pro default
                "topP": body.get("top_p", TOPP),  # Gemini 1.5 Pro default
                "topK": body.get("top_k", TOPK),  # A reasonable TopK for 1.5 Pro
                # "stopSequences": ["example_stop_sequence"] # Optional: Add stop sequences if needed
            },
            "safetySettings": safety_settings_config,  # Include safety settings
            # "tools": [] # Optional: Add tools if using function calling
        }

        # Remove stream parameter from generationConfig as it's handled by the endpoint choice
        stream = body.get("stream", False)

        if DEBUG:
            print("Gemini Request URL:", url)
            print("Gemini Request Data:", json.dumps(data, indent=2))

        if stream:
            # Use the streamGenerateContent endpoint logic
            return self._stream_gemini_response(url, data)
        else:
            # Use the generateContent endpoint logic
            response = self.authed_session.post(url, json=data)
            if response.status_code == 200:
                response_data = response.json()
                if DEBUG:
                    print("Gemini Response Data:", json.dumps(response_data, indent=2))

                # Extract text from the generateContent response structure
                try:
                    # Check for safety blocks first
                    if response_data.get("promptFeedback", {}).get("blockReason"):
                        reason = response_data["promptFeedback"]["blockReason"]
                        message = (
                            f"Request blocked due to prompt safety filters: {reason}"
                        )
                        if DEBUG:
                            print(message)
                        return f"Error: {message}"

                    if not response_data.get("candidates"):
                        if DEBUG:
                            print("No candidates found in response.")
                        return (
                            "Error: No response generated by the model (no candidates)."
                        )

                    first_candidate = response_data["candidates"][0]

                    if first_candidate.get("finishReason") not in [
                        "STOP",
                        "MAX_TOKENS",
                        None,
                    ]:  # None for cases where it might be missing but content is present
                        reason = first_candidate.get("finishReason", "Unknown")
                        message = f"Content generation stopped due to: {reason}"
                        safety_ratings = first_candidate.get("safetyRatings", [])
                        if safety_ratings:
                            message += f". Safety Ratings: {json.dumps(safety_ratings)}"
                        if DEBUG:
                            print(message)
                        # Return error if blocked, otherwise might return partial content if available
                        if reason == "SAFETY":
                            return f"Error: {message}"
                        # else: could potentially return partial content here if needed

                    if (
                        "content" in first_candidate
                        and "parts" in first_candidate["content"]
                    ):
                        generated_text = "".join(
                            part.get("text", "")
                            for part in first_candidate["content"]["parts"]
                        )
                        return generated_text
                    else:
                        if DEBUG:
                            print("No content.parts found in the first candidate.")
                        return "Error: No text content found in the model response."

                except (IndexError, KeyError, TypeError) as e:
                    if DEBUG:
                        print(f"Error parsing Gemini response: {e}")
                        print(
                            "Response Data received:",
                            json.dumps(response_data, indent=2),
                        )
                    return f"Error: Could not parse the response from the model. Structure might be unexpected. Details: {e}"
            else:
                error_message = f"Error: {response.status_code} {response.text}"
                if DEBUG:
                    print(error_message)
                    print("Response Headers:", response.headers)
                return error_message

    def _stream_gemini_response(self, url: str, data: dict) -> Iterator[str]:
        """Stream the response from the Gemini streamGenerateContent API."""
        response = self.authed_session.post(
            url, json=data, stream=True, headers={"Accept": "text/event-stream"}
        )

        if response.status_code != 200:
            error_message = f"Error: {response.status_code} {response.text}"
            if DEBUG:
                print(error_message)
                print("Response Headers:", response.headers)
            yield error_message  # Yield error and stop
            return

        if DEBUG:
            print("Starting to stream Gemini response...")

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    try:
                        # Remove "data: " prefix and parse JSON
                        content = decoded_line[len("data: ") :]
                        # Handle potential multi-chunk lines in SSE (though less common with Vertex)
                        if not content:
                            continue  # Skip empty data lines

                        json_response = json.loads(content)
                        if DEBUG:
                            print(
                                "Streaming Gemini chunk:",
                                json.dumps(json_response, indent=2),
                            )

                        # Check for prompt feedback block - happens early
                        if json_response.get("promptFeedback", {}).get("blockReason"):
                            reason = json_response["promptFeedback"]["blockReason"]
                            error_msg = f"Error: Request blocked due to prompt safety filters: {reason}"
                            if DEBUG:
                                print(error_msg)
                            yield error_msg
                            return  # Stop streaming if prompt is blocked

                        if (
                            "candidates" in json_response
                            and len(json_response["candidates"]) > 0
                        ):
                            candidate = json_response["candidates"][0]

                            # Check for finish reason indicating block
                            if candidate.get("finishReason") == "SAFETY":
                                ratings = candidate.get("safetyRatings", "N/A")
                                error_msg = f"Error: Content generation stopped due to safety filters. Ratings: {json.dumps(ratings)}"
                                if DEBUG:
                                    print(error_msg)
                                yield error_msg
                                # Potentially stop streaming here, or just yield the error and let it finish if other text came first? Yielding seems safer.
                                return  # Stop streaming if content is blocked

                            if (
                                "content" in candidate
                                and "parts" in candidate["content"]
                            ):
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        yield part["text"]
                    except json.JSONDecodeError as e:
                        if DEBUG:
                            print(
                                f"Error decoding JSON stream chunk: {e}, Line: '{decoded_line}'"
                            )
                        # Continue to next line might be okay, or yield error?
                        # yield f"Error decoding stream: {e}" # Option to notify client
                        continue
                    except Exception as e:
                        if DEBUG:
                            import traceback

                            print(
                                f"Error processing Gemini stream: {e}\n{traceback.format_exc()}"
                            )
                        # yield f"Error processing stream: {e}" # Option to notify client
                        continue  # Continue processing stream if possible
            # else: handle keep-alive or empty lines if necessary based on API behavior
        if DEBUG:
            print("Finished streaming Gemini response.")

    # Helper function to format messages for Gemini generateContent API
    def format_gemini_contents(self, messages):
        formatted_contents = []
        system_instruction = None  # Gemini specific: Separate system instruction

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if not content:  # Skip empty messages
                continue

            # Handle system messages - store the first one separately
            if role == "system":
                if system_instruction is None:  # Use the first system message
                    system_instruction = {"parts": [{"text": content}]}
                else:  # Append subsequent system messages as user messages (or log warning)
                    if DEBUG:
                        print(
                            "Warning: Multiple system messages found. Using first as system instruction, adding others as user messages."
                        )
                    formatted_contents.append(
                        {"role": "user", "parts": [{"text": f"System Note: {content}"}]}
                    )
                continue  # Move to next message after handling system message

            # Map "assistant" role to "model" for Gemini API
            api_role = "model" if role == "assistant" else "user"

            # Append user/assistant messages
            formatted_contents.append({"role": api_role, "parts": [{"text": content}]})

        # Prepend system instruction if it exists
        if system_instruction:
            # The generateContent API can take an explicit 'systemInstruction' field
            # For simplicity here, we keep it within 'contents' as the first 'user' message
            # Note: For optimal performance, using the dedicated system_instruction field
            # might be better if restructuring the call logic.
            # Adding it as the first element in contents - API usually handles turn order.
            # A dedicated `system_instruction` field outside 'contents' is preferred in v1beta1.
            # Let's structure payload later to include it correctly if needed.
            # For now, prepend as a user message for compatibility, though less ideal.
            # ***CORRECTION: Add system instruction correctly OUTSIDE contents if possible in payload***
            # Let's adjust the `data` dict in `call_gemini_model` instead.
            # For now, just return the formatted user/model contents.
            # Return value structure: { "contents": [...], "system_instruction": {...} } if needed later.
            pass  # System instruction handled separately in call_gemini_model if API supports it directly

        # Let's assume for now we will pass system instruction separately if possible
        # Or handle it by modifying the main data dict. Return just user/model turns here.
        final_formatted_contents = []
        current_role = None
        current_parts = []

        # Merge consecutive messages of the same role (required by Gemini API)
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if not content:
                continue

            if role == "system":
                # Handle system prompt - often best passed separately or as first user turn
                # Let's assume first system message becomes a user message for simplicity here
                if DEBUG:
                    print(
                        "Treating system message as the initial user message for content history."
                    )
                role = "user"  # Treat as user for content list

            api_role = "model" if role == "assistant" else "user"

            if api_role == current_role:
                current_parts.append({"text": content})
            else:
                # Finalize previous role block if it exists
                if current_role is not None:
                    final_formatted_contents.append(
                        {"role": current_role, "parts": current_parts}
                    )

                # Start new role block
                current_role = api_role
                current_parts = [{"text": content}]

        # Append the last block
        if current_role is not None:
            final_formatted_contents.append(
                {"role": current_role, "parts": current_parts}
            )

        return final_formatted_contents

    # --- Claude and Llama methods remain largely unchanged, but ensure auth and error handling are robust ---

    def call_claude_model(self, body, project_id, location, model_id):
        # Note: Claude on Vertex might use a different endpoint structure.
        # This uses the Anthropic publisher endpoint structure. Double-check documentation.
        # The model ID format might be just `claude-3-opus@20240229` etc. without publisher prefix for some calls.
        # Assuming the format requires the publisher:
        # url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:streamRawPredict?alt=sse" if body.get("stream", False) else f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:rawPredict"
        # Let's assume :predict endpoint for simplicity first, check if needs rawPredict/streamRawPredict
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:predict"

        stream = body.get(
            "stream", False
        )  # Claude :predict might not support streaming directly? TBC
        # Use :streamRawPredict for streaming Claude

        if stream:
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:streamRawPredict?alt=sse"
        else:
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:predict"
            # url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:rawPredict" # Use rawPredict for more control?

        messages = body["messages"]
        formatted_messages, system_prompt = self.format_claude_messages(messages)

        # Prepare request data - structure depends on rawPredict vs predict
        # Using the :predict structure (simpler, less control)
        # data_predict = {
        #     "instances": [{"prompt": formatted_prompt}], # Simple prompt string format
        #     "parameters": { ... }
        # }

        # Using rawPredict/streamRawPredict structure (matches Anthropic API closer)
        payload = {
            "anthropic_version": "vertex-2023-10-16",  # Or newer version if available/needed
            "messages": formatted_messages,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 1024),  # Claude often uses max_tokens
            "top_p": body.get("top_p", 0.8),
            "top_k": body.get("top_k", 40),
            "stream": stream,
            # Add system prompt if extracted
            **({"system": system_prompt} if system_prompt else {}),
        }

        # Need to wrap payload correctly for Vertex rawPredict/streamRawPredict
        data = {
            "contents": [json.dumps(payload)]
        }  # Wrap the JSON payload string inside 'contents' list for rawPredict

        if not stream:
            # For non-streaming :predict, structure is different:
            data = {
                "instances": [
                    {
                        "messages": formatted_messages,
                        # Add system prompt if extracted and supported by :predict instance structure
                        **({"system": system_prompt} if system_prompt else {}),
                    }
                ],
                "parameters": {
                    "anthropic_version": "vertex-2023-10-16",
                    "temperature": body.get("temperature", 0.7),
                    "maxOutputTokens": body.get("max_tokens", 1024),
                    "topP": body.get("top_p", 0.8),
                    "topK": body.get("top_k", 40),
                    # Stream parameter might not be valid for non-streaming :predict
                },
            }
            # Remove system from instance if it should be in parameters or is not supported by :predict
            if (
                "system" in data["instances"][0] and not False
            ):  # Check if :predict supports system in instance
                del data["instances"][0]["system"]
                data["parameters"][
                    "system"
                ] = system_prompt  # Try putting in parameters if applicable

        if DEBUG:
            print("Claude Request URL:", url)
            print("Claude Request Data:", json.dumps(data, indent=2))

        if stream:
            # Use streamRawPredict endpoint
            return self._stream_claude_raw_predict_response(
                url, {"contents": [json.dumps(payload)]}
            )  # Ensure payload is correct for streamRawPredict
        else:
            # Use predict endpoint
            response = self.authed_session.post(url, json=data)
            if response.status_code == 200:
                response_data = response.json()
                if DEBUG:
                    print("Claude Response Data:", json.dumps(response_data, indent=2))
                try:
                    # Parse response from :predict
                    if (
                        "predictions" in response_data
                        and len(response_data["predictions"]) > 0
                    ):
                        # Structure of prediction depends on underlying model (Claude)
                        # It might be {'content': [{'type': 'text', 'text': '...'}], ...}
                        prediction_content = response_data["predictions"][
                            0
                        ]  # Get the first prediction object

                        # Check for explicit content block based on safety
                        if prediction_content.get("stopReason") == "max_tokens":
                            if DEBUG:
                                print(
                                    "Warning: Output truncated due to max_tokens limit."
                                )
                        elif prediction_content.get("stopReason") not in [
                            None,
                            "stop_sequence",
                            "end_turn",
                        ]:
                            reason = prediction_content.get("stopReason", "Unknown")
                            # Claude safety stop reasons might differ, e.g., could be in a dedicated field or part of stopReason. Check docs.
                            # Assuming a non-standard stopReason might indicate an issue.
                            message = f"Content generation stopped potentially due to non-standard reason: {reason}"
                            if DEBUG:
                                print(message)
                            # Decide if this is an error or just informational
                            # return f"Error: {message}" # Example: treat as error

                        # Extract text content
                        generated_text = ""
                        if isinstance(prediction_content.get("content"), list):
                            for block in prediction_content["content"]:
                                if block.get("type") == "text":
                                    generated_text += block.get("text", "")
                            return generated_text
                        elif isinstance(
                            prediction_content.get("content"), str
                        ):  # Fallback if content is just a string
                            return prediction_content.get("content", "")
                        else:
                            if DEBUG:
                                print(
                                    "Could not find expected text content block in Claude prediction."
                                )
                            return "Error: No text content found in the model response."

                    else:
                        if DEBUG:
                            print("No predictions found in the Claude response.")
                        return "Error: No response generated by the model (no predictions)."
                except (IndexError, KeyError, TypeError) as e:
                    if DEBUG:
                        print(f"Error parsing Claude response: {e}")
                        print(
                            "Response Data received:",
                            json.dumps(response_data, indent=2),
                        )
                    return f"Error: Could not parse the response from Claude model. Details: {e}"
            else:
                error_message = f"Error: {response.status_code} {response.text}"
                if DEBUG:
                    print(error_message)
                    print("Response Headers:", response.headers)
                return error_message

    def _stream_claude_raw_predict_response(
        self, url: str, data: dict
    ) -> Iterator[str]:
        """Stream the response from the Claude streamRawPredict API."""
        response = self.authed_session.post(
            url, json=data, stream=True, headers={"Accept": "text/event-stream"}
        )

        if response.status_code != 200:
            error_message = f"Error: {response.status_code} {response.text}"
            if DEBUG:
                print(error_message)
                print("Response Headers:", response.headers)
            yield error_message
            return

        if DEBUG:
            print("Starting to stream Claude response...")

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    try:
                        content = decoded_line[len("data: ") :].strip()
                        if not content:
                            continue

                        # streamRawPredict returns chunks of the raw Anthropic response stream
                        # Each chunk needs to be parsed as JSON representing Anthropic stream events
                        json_chunk = json.loads(content)
                        if DEBUG:
                            print(
                                "Streaming Claude chunk:",
                                json.dumps(json_chunk, indent=2),
                            )

                        event_type = json_chunk.get("type")

                        if event_type == "content_block_delta":
                            delta = json_chunk.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                        elif event_type == "message_stop":
                            stop_reason = json_chunk.get("message", {}).get(
                                "stop_reason"
                            )
                            if DEBUG:
                                print(
                                    f"Claude stream finished. Stop Reason: {stop_reason}"
                                )
                            # Check if stop reason indicates an error or safety block
                            if stop_reason not in [
                                "end_turn",
                                "stop_sequence",
                                "max_tokens",
                            ]:
                                # Handle other reasons like 'tool_use' or potential error states if needed
                                warning_msg = f"Warning: Stream stopped due to reason: {stop_reason}"
                                if DEBUG:
                                    print(warning_msg)
                                # Decide if you need to yield an error message here.
                                # yield f"Error: {warning_msg}"
                            break  # Stop iteration cleanly on message_stop
                        elif event_type == "error":
                            error_data = json_chunk.get("error", {})
                            error_msg = f"Error from Claude stream: {error_data.get('type')} - {error_data.get('message')}"
                            if DEBUG:
                                print(error_msg)
                            yield error_msg
                            break  # Stop iteration on error

                        # Handle other event types like 'message_start', 'content_block_start', etc. if needed

                    except json.JSONDecodeError as e:
                        if DEBUG:
                            print(
                                f"Error decoding JSON stream chunk: {e}, Line: '{decoded_line}'"
                            )
                        continue
                    except Exception as e:
                        if DEBUG:
                            import traceback

                            print(
                                f"Error processing Claude stream: {e}\n{traceback.format_exc()}"
                            )
                        continue
        if DEBUG:
            print("Finished streaming Claude response.")

    def call_llama_model(self, body, project_id, location, model_id):
        # Placeholder for Llama model call
        # Llama models might be served via specific endpoints (e.g., tuned models)
        # or potentially the general openapi/chat/completions endpoint like Gemini
        # Need to determine the correct endpoint and payload structure for Llama on Vertex AI
        # Example using a hypothetical Llama endpoint or the generic one if applicable
        # url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/YOUR_LLAMA_ENDPOINT_ID:predict"
        # Or if using the chat completions compatible endpoint:
        # url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
        if DEBUG:
            print(
                f"Llama model call requested for {model_id}, but implementation is a placeholder."
            )
            print(
                "Determine the correct Vertex AI endpoint and payload structure for Llama models."
            )

        # --- You would need to adapt the following based on the actual Llama endpoint ---
        # Example assuming a compatible chat completions endpoint:
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
        stream = body.get("stream", False)
        messages = body["messages"]
        formatted_messages = self.format_messages(messages)  # Use generic OA format

        # Construct payload (might need adjustments for Llama specifics)
        # Note: Llama model name likely needs publisher prefix, e.g., "meta/llama3-8b-instruct"
        # Find the correct ID format in Vertex AI Model Garden / Registry
        llama_formatted_model_id = f"meta/{model_id}"  # Example, adjust as needed
        data = {
            "model": llama_formatted_model_id,
            "messages": formatted_messages,
            "temperature": body.get("temperature", 0.6),  # Llama defaults might differ
            "max_tokens": body.get("max_tokens", 1024),
            "top_p": body.get("top_p", 0.9),
            "top_k": body.get("top_k", 50),
            "stream": stream,
            # Safety settings might not be supported or use different structure for Llama endpoints
        }

        if DEBUG:
            print("Hypothetical Llama Request URL:", url)
            print("Hypothetical Llama Request Data:", json.dumps(data, indent=2))

        if stream:
            # Use the generic _stream_response if endpoint supports OpenAI SSE format
            return self._stream_response(url, data)
        else:
            # Use generic POST request logic
            response = self.authed_session.post(url, json=data)
            if response.status_code == 200:
                response_data = response.json()
                if DEBUG:
                    print("Llama Response Data:", json.dumps(response_data, indent=2))
                # Parse response (assuming OpenAI compatible structure)
                try:
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        generated_text = (
                            response_data["choices"][0]
                            .get("message", {})
                            .get("content", "")
                        )
                        # Check finish reason for potential issues (e.g., length, safety)
                        finish_reason = response_data["choices"][0].get("finish_reason")
                        if finish_reason not in ["stop", None]:
                            print(
                                f"Warning: Llama generation finished due to: {finish_reason}"
                            )
                            if finish_reason == "length":
                                # Optionally append a note about truncation
                                generated_text += (
                                    "\n[Output truncated due to length limit]"
                                )
                            elif finish_reason == "safety":
                                return f"Error: Response blocked due to safety settings."  # Or return partial + error
                        return generated_text
                    else:
                        if DEBUG:
                            print("No choices found in Llama response.")
                        return (
                            "Error: No response generated by Llama model (no choices)."
                        )
                except (IndexError, KeyError, TypeError) as e:
                    if DEBUG:
                        print(f"Error parsing Llama response: {e}")
                    return f"Error: Could not parse Llama response. Details: {e}"

            else:
                error_message = f"Error: {response.status_code} {response.text}"
                if DEBUG:
                    print(error_message)
                return error_message

        # return "Error: Llama model support not fully implemented yet."

    # Generic formatter (OpenAI compatible structure)
    def format_messages(self, messages):
        formatted_messages = []
        # The chat completions endpoint expects alternating user/assistant roles
        # System message is often handled differently (first message or dedicated field)
        system_prompt = None
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if not content:
                continue

            if role == "system" and system_prompt is None:
                system_prompt = content  # Capture first system prompt
                # How it's used depends on the API, often placed first
            elif role == "user":
                formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
            elif role == "system":  # Handle subsequent system messages if any
                if DEBUG:
                    print(
                        "Warning: Multiple system messages. Adding subsequent ones as user messages."
                    )
                formatted_messages.append(
                    {"role": "user", "content": f"System Note: {content}"}
                )

        # Prepend system prompt as the first message if captured
        if system_prompt:
            # Check if last message was assistant, if so, can't prepend system (must alternate)
            # This simple prepend assumes it's okay or first message. Real logic might need merging/adjusting turns.
            formatted_messages.insert(
                0, {"role": "system", "content": system_prompt}
            )  # Use 'system' role if API supports it

        # Ensure correct turn order (user, assistant, user, ...) - API might enforce this
        # Basic check: No two consecutive messages of the same role
        final_messages = []
        last_role = None
        for msg in formatted_messages:
            current_role = msg["role"]
            # Simple merge strategy: Combine content if same role follows (basic)
            # A more robust strategy might be needed depending on API strictness
            if final_messages and current_role == last_role:
                if DEBUG:
                    print(f"Warning: Merging consecutive {current_role} messages.")
                final_messages[-1]["content"] += "\n" + msg["content"]
            else:
                final_messages.append(msg)
                last_role = current_role

        return final_messages

    def format_claude_messages(self, messages):
        # Claude API expects format: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        # System prompts handled via a separate 'system' parameter in the main request body
        formatted_messages = []
        system_prompt = None
        last_role = None

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")

            if not content:
                continue  # Skip empty content

            if role == "system":
                if system_prompt is None:  # Capture the first system message
                    system_prompt = content
                else:  # Append additional system messages to the main one
                    system_prompt += "\n" + content
                continue  # Don't add system messages to the main message list

            # Ensure alternating user/assistant roles, starting with user
            # If the first non-system message is 'assistant', prepend a dummy user message if required by API
            if not formatted_messages and role == "assistant":
                if DEBUG:
                    print(
                        "Warning: First message is 'assistant'. Claude API requires starting with 'user'. Prepending empty user message (consider fixing history)."
                    )
                # formatted_messages.append({"role": "user", "content": "[Initiating assistant response]"}) # Or handle error
                # Best practice: Ensure history always starts with user

            # Merge consecutive messages of same role OR handle error
            if role == last_role:
                if DEBUG:
                    print(
                        f"Warning: Merging consecutive '{role}' messages for Claude API compatibility."
                    )
                if formatted_messages:  # Ensure list is not empty
                    formatted_messages[-1]["content"] += "\n" + content
                else:  # Should not happen if logic is correct, but handle nonetheless
                    formatted_messages.append({"role": role, "content": content})
                continue  # Skip appending separately

            # Add the message
            formatted_messages.append({"role": role, "content": content})
            last_role = role

        # Final check: Ensure last message is not assistant if expecting user input next? Typically handled by caller.

        return (
            formatted_messages,
            system_prompt,
        )  # Return both messages and system prompt

    # Generic stream response parser (OpenAI compatible SSE)
    # Used by call_gemini_model (if using openapi endpoint) and potentially call_llama_model
    def _stream_response(self, url: str, data: dict) -> Iterator[str]:
        """Stream the response from an OpenAI-compatible SSE endpoint."""
        response = self.authed_session.post(
            url, json=data, stream=True, headers={"Accept": "text/event-stream"}
        )

        if response.status_code != 200:
            error_message = f"Error: {response.status_code} {response.text}"
            if DEBUG:
                print(error_message)
                print("Response Headers:", response.headers)
            yield error_message
            return

        if DEBUG:
            print(f"Starting to stream generic SSE response from {url}")

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    content = decoded_line[len("data: ") :]
                    if content.strip() == "[DONE]":
                        if DEBUG:
                            print("SSE stream finished with [DONE].")
                        break  # End of stream marker
                    try:
                        json_response = json.loads(content)
                        if DEBUG:
                            print(
                                "Streaming generic chunk:",
                                json.dumps(json_response, indent=2),
                            )

                        if (
                            "choices" in json_response
                            and len(json_response["choices"]) > 0
                        ):
                            delta = json_response["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"] is not None:
                                yield delta["content"]
                            # Check for finish reason in the last chunk if needed
                            # finish_reason = json_response['choices'][0].get('finish_reason')
                            # if finish_reason:
                            #     if DEBUG: print(f"Stream finished with reason: {finish_reason}")
                            #     # Handle finish reason (e.g. safety, length) if necessary
                            #     if finish_reason == "safety": yield "\n[Error: Response stopped due to safety filter]"
                            #     elif finish_reason == "length": yield "\n[Warning: Response truncated due to length]"
                            #     break # Stop on finish reason

                    except json.JSONDecodeError as e:
                        if DEBUG:
                            print(
                                f"Error decoding JSON stream chunk: {e}, Line: '{decoded_line}'"
                            )
                        continue
                    except Exception as e:
                        if DEBUG:
                            import traceback

                            print(
                                f"Error processing generic stream: {e}\n{traceback.format_exc()}"
                            )
                        continue
                elif (
                    decoded_line.strip()
                ):  # Handle non-data lines if needed (e.g., error messages not in JSON)
                    if DEBUG:
                        print(f"Received non-data line: {decoded_line}")

        if DEBUG:
            print(f"Finished streaming generic SSE response from {url}")
