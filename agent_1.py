from dotenv import load_dotenv
import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

def encode_image(image_path):
    """Encode an image file to base64."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class HomeInspectionAgent(Runnable[dict, dict]):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("user_input", "")
        image_path = inputs.get("image_path", None)

        # Base message structure
        system_message = SystemMessage(content=(
            "You are a professional virtual Home Inspection expert. Your job is to carefully analyze the user's description "
            "and the attached image (if provided) to detect visible household issues, such as water damage, mold, cracks, "
            "wiring problems, or plumbing issues.\n\n"
            "Provide your diagnosis in simple language and give a specific next step recommendation (e.g., contact a plumber, "
            "electrician, or perform DIY fix).\n"
            "If there is no visible issue or it's unclear, politely say that the image doesn’t show any major concern and suggest "
            "next steps if needed.\n\n"
            "Format:\n"
            "- Issue Identified: <description>\n"
            "- Recommendation: <next step>\n\n"
            "Example:\n"
            "Issue Identified: The pipe joint appears to be leaking slightly.\n"
            "Recommendation: Call a certified plumber to reseal or replace the joint."
            "use memory to understand context of each new query"
            "memory: {memory}"
        ))

        # Construct HumanMessage based on whether image is provided
        if image_path:
            try:
                base64_image = encode_image(image_path)
                human_message = HumanMessage(content=[
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ])
            except FileNotFoundError as e:
                return {"error": str(e)}
        else:
            human_message = HumanMessage(content=user_input)

        response = self.llm.invoke([system_message, human_message])
        return {"output": response.content}

# Create agent instance
home_inspection_agent = HomeInspectionAgent(llm=llm)


