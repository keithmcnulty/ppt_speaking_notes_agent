import base64
import io
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os
import pandas as pd
from pdf2image import convert_from_path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration - Update these with your values
OPENAI_API_BASE_URL = os.getenv("OPENAI_BASE_URL")  # Update if using a proxy
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Your OpenAI API key
OPENAI_VISION_MODEL = "gpt-4o"

# Initialize clients
openai_client = OpenAI(
    base_url=OPENAI_API_BASE_URL,
    api_key=OPENAI_API_KEY,
)

class SpeakingNotes(BaseModel):
    SlideNo: int = Field(..., description="Number of the slide")
    SpeakingNotes: str = Field(..., description="Speaking notes for the slide") 
    FollowUpQns: str = Field(..., description="Follow-up questions and suggested responses for the slide")

def png_file_to_base64(path: str) -> str:
    """Convert a local PNG file to a base64-encoded string using Pillow."""
    with Image.open(path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

async def analyze_with_openai_vision(file:str, prompt: str) -> str:
    """Analyze image using OpenAI GPT-4 Vision."""
    base64_image = png_file_to_base64(file)
    
    response = openai_client.beta.chat.completions.parse(
        model=OPENAI_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            },
        ],
        max_tokens=16000,
        response_format = {"type": "json_object"},
    )
    return response.choices[0].message.content

async def create_notes(image: str, audience: str, context: str,
                       model: str = "gpt-4o") -> pd.DataFrame:
    """
    Create speaking notes for a presentation based on the provided image of slides.

    Args:
        image (str): Path to the image file containing the slides.
        audience (str): Description of the intended audience for the presentation.
        context (str): Context or background information relevant to the presentation.
        model (str): The OpenAI model to use for analysis (default is "gpt-4o").        
    Returns:
        pd.DataFrame: A DataFrame containing the slide number, speaking notes, and follow-up
        questions.
    """
    
    # Step 1: Create analysis prompt
    prompt = """
    The image contains several consecutive slides from a presentation.

    The presentation is intended to be presented to an audience as follows:  {audience}.  
    Here is the context of the presentation: {context}.

    Your task is to analyze the image and create comprehensive speaking notes for the presenter.
    The notes should be detailed, covering all key points, explanations, and any relevant context
    that would help the presenter effectively communicate the content of the slides.  Provide a strong level of background detail
    about each topic covered in the slides, ensuring that the presenter can deliver the content confidently.
    Also suggest up to three folow up questions that the audience might ask for each slide, and advise how the presenter should respond to them.
    Do not encapsulate the folllow up questions in a list, but rather provide them as a narrative.
    The notes should be structured in a way that allows the presenter to easily follow along
    during the presentation, providing a clear narrative flow.
    Please ensure that the notes are concise yet informative, avoiding unnecessary jargon
    while still being accurate. The notes should also consider the audience's
    background and be engaging the audience.


    Number the speaking notes and follow-up questions according to the slide numbers which are clearly directly beneath each slide.

    Return a JSON object with a key of `results` and a value which is a list of JSON objects, each containing the slide number under the key `SlideNo`, the corresponding speaking notes
    under the key `SpeakingNotes`, and the potential follow up questions and respinses under the key `FollowUpQns`. For example:

    {"results" : [{"SlideNo": 1, "SpeakingNotes": "some speaking notes for slide 1", "FollowUpQns": "Some follow-up questions for Slide 1 and suggested responses"}, 
    {"SlideNo": 2, "SpeakingNotes": "Some speaking notes for slide 2", "FollowUpQns": "Some follow-up questions for Slide 1 and suggested responses"}, ...]} 
    """
    

    # Step 2: Call vision model (using OpenAI as example)
    try:
        analysis = await analyze_with_openai_vision(image, prompt)
        result = json.loads(analysis)
        docs = result.get('results')
        new_docs = []
        for i, doc in enumerate(docs):
            new_docs.append(SpeakingNotes(**doc))
        return pd.DataFrame([res.model_dump() for res in new_docs]).sort_values(by='SlideNo').reset_index(drop=True)
        
    except Exception as e:
        print(f"OpenAI Vision failed: {e}")
        # Fallback to basic analysis
        return "Visual analysis unavailable. Please check API configuration." 


async def speaker_notes(file: str, audience: str, context: str, output_file: str,
                        model: str = "gpt-4o") -> None:
    """
    Generate speaking notes from a PDF file containing slides.

    Args:
        file (str): Path to the PDF file.
        audience (str): Description of the intended audience for the presentation.
        context (str): Context or background information relevant to the presentation.
        model (str): The OpenAI model to use for analysis (default is "gpt-4o").

    Returns:
        pd.DataFrame: A DataFrame containing the slide number, speaking notes, and follow-up questions.
    """

    base = os.path.basename(file)
    name, ext = os.path.splitext(base)
    if ext.lower() != '.pdf':
        raise ValueError("The file must be a PDF.")
    # Check whether the specified path exists or not
    isExist = os.path.exists(name)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(name)


    # Store pdf page images with convert_from_path function
    images = convert_from_path(file, dpi=300)

    for i in range(len(images)):
    
        # Save pages as images in the pdf
        images[i].save(os.path.join(name, 'page'+ str(i) +'.png'), 'PNG')
    
    results = []
    i = 0
    for file in os.listdir(name):
        if file.endswith(".png"):
            i += 1
            logger.info(f"Processing page {i} of {len(images)}...")
            speaking_notes = await create_notes(
                image = os.path.join(name, file),    
                audience = audience,
                context = context,
                model = model
            )
            results.append(speaking_notes) 
    
    # Delete the directory after processing
    for file in os.listdir(name):
        os.remove(os.path.join(name, file))
    os.rmdir(name)
    if results:
        logger.info("All slides processed and speaking notes generated.")
        # Concatenate all results into a single DataFrame
        pd.concat(results, ignore_index=True).sort_values(by='SlideNo').reset_index(drop=True).to_csv(output_file, index=False)
    else:    
        logger.warning("No speaking notes generated. Please check the input file and try again.")


    
    
