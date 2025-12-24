import gradio as gr
from transformers import pipeline
import openai
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from dotenv import load_dotenv

title = "Ambience23"

load_dotenv()

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
gpt4 = ChatOpenAI(model='gpt-4', temperature=0.2, streaming=True, verbose=True)
notes=""

clinical_note_writer_template = PromptTemplate(
    input_variables=["transcript"],
    template=
    """Based on the conversation transcript provided below, generate a clinical note in the following format:
Diagnosis:
History of Presenting Illness:
Medications (Prescribed): List current medications and note if they are being continued, or if any new ones have been added.
Lab Tests (Ordered):
Please consider any information in the transcript that might be relevant to each of these sections.

Please refer to following for example:

Diagnosis: Uncontrolled Diabetes and Hypertension
History of Presenting Illness: The patient has been adhering to their current medication regimen but the diabetes and hypertension seem uncontrolled.
Medications (Prescribed):
[Continue] Glycomet-GP 1 (tablet) | Glimepiride and Metformin
[Added] Jalra-OD 100mg (tablet) | Vildagliptin
[Added] Telmis 20 (Tablet)
Lab Tests (Ordered): HbA1c (Glycated Hemoglobin), Urinalysis (including microalbuminuria), Postprandial Blood Glucose, Blood Urea Nitrogen (BUN)
Now, based on the following conversation and hints, please generate a clinical note:

### Conversation Transcript
{transcript}
""")

def consultation_transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    transcript = transcriber({"sampling_rate": sr, "raw": stream})["text"]
    global notes
    notes = LLMChain(llm=gpt4, prompt=clinical_note_writer_template, verbose=True).run({"transcript": transcript})

    return stream, transcript, notes

patient_chat_template = PromptTemplate(
    input_variables=["question", "notes"],
    template=
    """As a medical chatbot, your task is to answer patient questions about their prescriptions. You should provide complete, scientifically-grounded, and actionable answers to queries, based on the provided recent clinical note.
You can communicate fluently in the patient's language of choice, such as English, French, Spanish, Arabic, Hindi etc. If the patient asks a question unrelated to the diagnosis or medications in the clinical note, your response should be, 'This question can't be answered on the basis of your recent consultation.'

### Recent Prescription
{notes}

Let's begin the conversation:
Patient: {question}
Bot:""")

def patientchat(question):
    sr, y = question
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    patient_question = transcriber({"sampling_rate": sr, "raw": y})["text"]
    answers_to_questions = LLMChain(llm=gpt4, prompt=patient_chat_template, verbose=True).run({"notes": notes, "question": patient_question})
    return answers_to_questions

doctor_ui = gr.Interface(
    fn=consultation_transcribe,
    inputs=["state", gr.Audio(sources=["microphone"], streaming=True, label="Speech")],
    outputs=["state", gr.Textbox(label="Transcript"), gr.Textbox(label="Notes")],
    live=True,
    title=title,
    description="Consultation",
)

patient_ui = gr.Interface(
    fn=patientchat,
    inputs=[gr.Audio(sources=["microphone"], label="Speech")],
    outputs=[gr.Textbox(label="Answers")],
    title=title,
    description="Questions?",
)

ui = gr.TabbedInterface([doctor_ui, patient_ui], ["Doctor", "Patient"])
ui.launch()