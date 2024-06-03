import chainlit as cl
import vertexai
import os
from vertexai.preview import reasoning_engines
from typing import Optional
from io import BytesIO
import pdfplumber
import librosa
import numpy as np
import soundfile as sf
import io
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import which
from transformers import pipeline
import requests
from PIL import Image
import aiohttp
from chainlit.input_widget import Switch
from config import API_URL, HF_API_KEY
headers = {"Authorization": HF_API_KEY} 
from chainlit.input_widget import TextInput
from audio_processing import process_audio
from vertex_agent import call_dialogflow, write_request_data
from image_processing import recognize_image
from tts import text_to_speech
from chainlit.input_widget import Select
import pdfplumber
import uuid 
from codeforces import latex_to_text,markdown_to_text
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from config import claude_api_key
import os
import json
from rag_SDE import get_sde_questions
from rag_ML import get_ml_questions
import requests
import hashlib
import time
import random
import string
from bs4 import BeautifulSoup
from markdown import markdown
import re
import latex2mathml.converter
from config import cf_api_key,cf_secret
import re

def remove_tags(text):
    # Define regex pattern to match anything within angle brackets
    pattern = re.compile(r'<[^>]+>')
    
    # Substitute the tags with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

class CodeforcesAPI:
    def __init__(self, api_key=cf_api_key, secret=cf_secret):
        self.api_key = api_key
        self.secret = secret
        self.base_url = "https://codeforces.com/api/"
      
    
    def _generate_api_sig(self, method_name, params):
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        current_time = int(time.time())
        params['apiKey'] = self.api_key
        params['time'] = current_time
        param_str = '&'.join(f"{key}={value}" for key, value in sorted(params.items()))
        hash_str = f"{rand}/{method_name}?{param_str}#{self.secret}"
        hash_hex = hashlib.sha512(hash_str.encode('utf-8')).hexdigest()
        return f"{rand}{hash_hex}"
    
    def _make_request(self, method_name, params):
        if self.api_key and self.secret:
            api_sig = self._generate_api_sig(method_name, params)
            params['apiSig'] = api_sig
        url = f"{self.base_url}{method_name}"
        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] == "OK":
            return data['result']
        else:
            raise Exception(f"API request failed: {data.get('comment', 'No comment provided')}")

    def user_status(self, handle):
        params = {'handle': handle, 'from': 1}
        return self._make_request("user.status", params)
    
    def unsolved_problems_by_ratings(self, handle, ratings):
        user_submissions = self.user_status(handle)
        solved_problems = {f"{sub['problem']['contestId']}{sub['problem']['index']}" for sub in user_submissions if sub['verdict'] == 'OK' and 'rating' in sub['problem']}
        method_name = "problemset.problems"
        params = {}
        all_problems = self._make_request(method_name, params)['problems']
        
        results = []
        for rating in ratings:
            unsolved = [{
                'contestId': p['contestId'],
                'index': p['index'],
                'name': p['name'],
                'rating': p['rating'],
                'url': f"https://codeforces.com/contest/{p['contestId']}/problem/{p['index']}"
            } for p in all_problems if f"{p['contestId']}{p['index']}" not in solved_problems and p.get('rating') == rating]
            
            if unsolved:
                results.append(random.choice(unsolved))
            else:
                results.append(f"No unsolved problems available for rating {rating}")
        
        return results
    
    def check_problems_solved(self, handle, problems):
        submissions = self.user_status(handle)
        solved_problems = {f"{sub['problem']['contestId']}{sub['problem']['index']}": sub['verdict'] for sub in submissions if sub['verdict'] == 'OK'}
        results = {}
        for problem in problems:
            problem_id = f"{problem['contestId']}{problem['index']}"
            results[problem_id] = 'Solved' if problem_id in solved_problems else 'Not Solved'
        return results
    
    def get_problem_description(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            problem_statement = soup.find('div', class_='problem-statement')
            if problem_statement:
                return problem_statement.get_text('\n', strip=True)
            else:
                return "Problem statement not found."
        except AttributeError:
            return "Problem statement not found."
        
    def check_problems_solved(self, handle, problems):
        submissions = self.user_status(handle)
        solved_problems = {f"{sub['problem']['contestId']}{sub['problem']['index']}": sub['verdict'] for sub in submissions if sub['verdict'] == 'OK'}
        results = {}
        for problem in problems:
            problem_id = f"{problem['contestId']}{problem['index']}"
            results[problem_id] = 'Solved' if problem_id in solved_problems else 'Not Solved'
        return results

    def user_info(self, handle):
        method_name = "user.info"
        params = {'handles': handle}
        try:
            self._make_request(method_name, params)
            return True  # The handle is valid
        except Exception:
            return False  # The handle is not valid


sde_questions_prompt ='''
You are an agent specialising in providing questions .Based on the information provided, generate a list of 8 high quality theoretical questions that would be suitable to ask a software developer candidate to test their understanding. The questions should be detailed and grammatically correct.

Some acceptable topics to cover in the questions include:
- Object-oriented programming concepts 
- Database management systems
- Computer architecture
- Software design patterns
- Algorithms and data structures
- Version control systems
- Web development frameworks
- Software testing methodologies

Make sure the questions are varied and cover different aspects of software development.

Please output the questions in a numbered list format, like this:

<questions>
1. [First question CV based]
2. [Second question general ]
3. [Third question  general ]
4. [Fourth question  general] 
5. [Fifth question CV based]
6. [Sixth question  general]
7. [Seventh question general]
8. [Eighth question general]
</questions>

Generate the 6 questions now and then using the CV text below generate 2 questions that test indepth knoledge of the user , they can be based on anything
such as technologies used in the projects on CV or any skills written in the CV. Make sure to generate a total of 8 questions!!
<CV_TEXT>

''' 
ml_questions_prompt = '''
You are an agent specialising in providing questions. Based on the information provided, generate a list of 8 high quality theoretical questions that would be relevant to ask a candidate for this type of machine learning role. 

The questions should be detailed, grammatically correct, and designed to test the depth of the candidate's understanding of key machine learning concepts. Some acceptable topics to ask about include:

- Traditional machine learning algorithms 
- Neural networks and deep learning
- Important mathematical and statistical concepts used in machine learning
- Model training, evaluation and optimization techniques
- Handling of data - preprocessing, feature engineering, etc.

Output your list of questions in the following format:

<questions>
1. [First question CV based]
2. [Second question general ]
3. [Third question  general ]
4. [Fourth question  general] 
5. [Fifth question CV based]
6. [Sixth question  general]
7. [Seventh question general]
8. [Eighth question general]
</questions>

Generate the 6 questions now and then using the CV text below generate 2 questions that test indepth knoledge of the user , they can be based on anything
such as technologies used in the projects on CV or any skills written in the CV. Make sure to generate a total of 8 questions!!
<CV_TEXT>
'''

tokenizer = Anthropic().tokenizer
Settings.tokenizer = tokenizer
async def evaluate_interview(value, claude_api_key=claude_api_key):
    
    evaluator_prompt = f'''
    You will be acting as an evaluator assessing a candidate's performance in an interview based on the transcript provided. The transcript will be in list of JSON format, contained within the following tags:

    <interview_transcript>
    {value}
    </interview_transcript>

    Parse the list of JSON to extract the interviewer's questions and the candidate's responses. For each question-response pair, evaluate the candidate's answer based on the following criteria:
    - Relevance: Does the answer directly address the question asked?
    - Clarity: Is the answer clear, concise, and easy to understand?
    - Depth of Knowledge: Does the answer demonstrate a strong understanding of the subject matter?
    - Specific Examples: Does the candidate provide specific examples to support their points?

    For each criterion, assign a score between 1 and 5, with 1 being poor and 5 being excellent. 

    Before providing the score for each response, write a brief justification for your scores inside <justification> tags, considering the above criteria and any other relevant factors. After the justification, provide the score inside <score> tags.

    After evaluating each individual response, provide an overall assessment of the candidate's performance in the interview. Consider factors such as:
    - Consistency: Did the candidate perform well throughout the interview, or were there significant variations in the quality of their responses?
    - Communication Skills: Did the candidate express themselves effectively and professionally?
    - Fit for the Role: Based on their responses, does the candidate seem well-suited for the position they are interviewing for?

    Write your overall assessment inside <overall_assessment> tags. Following the assessment, assign an overall interview score between 1 and 5 inside <overall_score> tags.

    Format your response as follows:

    <evaluation>
    <question_1>question 1 repeated here</question_1>
    <candidate_answer>candidate answer to question 1 repeated here </candidate_answer>
    <question_1_evaluation>
    <justification>Your justification for the scores</justification>
    <score>Your scores for question 1</score>
    </question_1_evaluation>

    <question_2_evaluation>
    <question_2>question 2 repeated here</question_2>
    <candidate_answer>candidate answer to question 1 repeated here </candidate_answer>
    <justification>Your justification for the scores</justification>
    <score>Your scores for question 2</score>
    </question_2_evaluation>

    ...

    <overall_assessment>Your overall assessment of the candidate's performance</overall_assessment>
    <overall_score>Your overall score for the candidate</overall_score>
    </evaluation>
    finally give your output in a neat format with proper formatting that is readable to the user
    Dont output the tags just make the tag text bold as headings and always give a clear tabular 
    format of the scores recieved at every step and in the end , make the output more concise,
    dont display tags in the final output
    '''

    os.environ["ANTHROPIC_API_KEY"] = claude_api_key

    messages = [
        ChatMessage(
            role="system", content=evaluator_prompt
        ),
        ChatMessage(role="user", content="Evaluate my interview responses"),
    ]
    resp = Anthropic(model="claude-3-sonnet-20240229").chat(messages, max_tokens=2048)

    return resp

class QuestionAnswerLogger:
    def __init__(self):
        self.data = []  # This list will store all question-answer pairs

    def add_question_answer(self, question, answer):
        self.data.append(
            {
            'question': {question},
            'answer': {answer}
            }
        )

    def get_all_entries(self):
        return self.data

    def write_to_json(self, file_path):
        """Write the question-answer data to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    return [
        cl.ChatProfile(name="Vertex AI ", markdown_description="The underlying LLM model is Vertex AI"),
        cl.ChatProfile(name="Interview Room", markdown_description="Interview Room"),
    ]


@cl.on_chat_start
async def on_chat_start():

    await cl.Avatar(
        name="Vertex AI",
        url="/public/leaf.png",
    ).send()

    session_id =  uuid.uuid1()
    # Set session variables to default values
    cl.user_session.set("session_id",str(session_id))
    cl.user_session.set("counter", 0)
    cl.user_session.set("user_role",None)
    cl.user_session.set("play_audios",False)
    cl.user_session.set("user_name",None)
    cl.user_session.set("cfid",None)
    cl.user_session.set("start_interview",False)
    cl.user_session.set("cv_text",None)
    cl.user_session.set("QA",None)
    cfobj = CodeforcesAPI()
    cl.user_session.set("CFOBJ",cfobj)

    chat_profile = cl.user_session.get("chat_profile", "Vertex AI")

    settings = await cl.ChatSettings(
        [
            Switch(id="play_audios", label="Enable Audio Playback", initial=False, tooltip ="Stop or play audio"),
        ]
    ).send()

    cl.user_session.set("play_audios", settings["play_audios"])
    

    if(chat_profile=='Interview Room'):

        res = await cl.AskUserMessage(author ="Vertex AI",content="Hi welcome to Interview Room!! \n Let's get started with some basic information. \n\n What is your name?").send()
        cl.user_session.set("user_name",res)
         
        res =  await cl.AskUserMessage(author ="Vertex AI",content="What is your Codeforces ID?").send()
     
        count = 0
        while(cfobj.user_info(str(res['output']))== False):
            res =  await cl.AskUserMessage(author ="Vertex AI",content="Entered ID is wrong please enter a valid ID !!").send()
            count+=1
            if(count>5):
                await cl.AskUserMessage(author ="Vertex AI",content="Too many attempts!! Please try later").send()
                return
            
        cl.user_session.set("cfid",res['output'])

        res = await cl.AskActionMessage(
        author ="Vertex AI",content="What is your prefered role?",
        actions=[
            cl.Action(name="SDE", value="SDE", label="Software Developer 🖥️"),
            cl.Action(name="ML", value="ML", label="Machine Learning 🤖"),
        ],
       ).send()
        cl.user_session.set("user_role",res)

        files = None
        while files == None:
            files = await cl.AskFileMessage(
                author ="Vertex AI",content="Please upload your Resume to begin!", accept=["application/pdf"]
            ).send()

        text_file = files[0]

        with pdfplumber.open(text_file.path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        cl.user_session.set("cv_text",text)

        start = await cl.AskActionMessage(
        author ="Vertex AI",content="Are you ready to start ?",
        actions=[
            cl.Action(name="Yes ", value="YES", label="Yes, I'm ready"),
            cl.Action(name="No", value="NO", label="Not now"),
        ],
        ).send()

        cl.user_session.set("start_interview",start)
        USER_NAME  = cl.user_session.get("user_name")['output']
        USER_ROLE  = cl.user_session.get("user_role")['value']
        CV_TEXT = cl.user_session.get("cv_text")
        CFID = cl.user_session.get("cfid")

        # if start interview is set to interview
        if(cl.user_session.get("start_interview")):

            user_init = f'''
            <User Info> CANDIDATE_NAME: {USER_NAME}
            CFID: {CFID} 
            USER_ROLE: {USER_ROLE} </User Info>
            <CV_INFO> {CV_TEXT} </CV_INFO>
            USER QUERY: Could you help me with preparing for an interview?
            '''
        
            QUESTIONS = " "
            if(USER_ROLE == "SDE"):
                QUESTIONS = await get_sde_questions(sde_questions_prompt+f"\n {CV_TEXT} \n </CV_TEXT>")
            
            elif(USER_ROLE == "ML"):
                QUESTIONS = await get_ml_questions(ml_questions_prompt+f"\n {CV_TEXT} \n </CV_TEXT>")
            
            # CODING PROBLEMS
            coding_problems = cfobj.unsolved_problems_by_ratings(CFID,[1600,1800])
            cl.user_session.set("cf_problems_list",coding_problems)
            probs = []
            submit_urls = []
            for problem in  coding_problems:
                if isinstance(problem, dict):
                    description = cfobj.get_problem_description(problem['url'])
                    plain_text_description = markdown_to_text(description)
                    # Ensure proper formatting for Input, Output, and Examples sections
                    formatted_description = re.sub(r'(?i)(Input|Output|Examples)', r'\n\1\n', plain_text_description)
                    probs.append(f"{problem['name']} \nDescription:\n{formatted_description}\n\n")
                    submit_urls.append(problem['url'])
                else:
                    probs.append(None)
                    submit_urls.append(None)

            cl.user_session.set("probs",probs)
            cl.user_session.set("submit_urls",submit_urls)
                
            user_information = f'''
            <User Info> CANDIDATE_NAME: {USER_NAME}
            CFID: {CFID} 
            USER_ROLE: {USER_ROLE} </User Info>
            <CV_INFO> {CV_TEXT} </CV_INFO>
            <INTERVIEW QUESTIONS BEGIN>
            {QUESTIONS}
            <INTERVIEW QUESTIONS END>
            USER QUERY: Could you help me with preparing for an interview?
            '''        
            msg  = cl.Message(content ="" , author= "Vertex AI")
            await msg.send()
            await cl.sleep(2)

            write_request_data(user_init)
            await call_dialogflow(cl.user_session.get("session_id"))
            write_request_data(user_information)
            response  = await call_dialogflow(cl.user_session.get("session_id"))
            if response and 'queryResult' in response:
                msg.content = response['queryResult']['responseMessages'][0]['text']['text'][0]
                await msg.update()
                cl.user_session.set("last_response",response['queryResult']['responseMessages'][0]['text']['text'][0])
            qa_logger = QuestionAnswerLogger()
            cl.user_session.set("QA",qa_logger)

        else:
           return 
        
@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("play_audios", settings["play_audios"])    

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.on_audio_end
async def on_audio_end(elements):
    image_results = []
    pdf_question= []
    image_available = False
    pdf_available = False
    play_audios = cl.user_session.get("play_audios")
    msg = cl.Message(content="",author="Vertex AI")
    await msg.send()
    await cl.sleep(2)
    audio_buffer = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0) 
    mime_type = cl.user_session.get("audio_mime_type")
    (user_question,transcript)  = process_audio(audio_buffer,mime_type)
    chat_profile = cl.user_session.get("chat_profile")
    if elements:  
        for file in elements:
            if file.path is not None:
                if "image" in file.mime: 
                    image_available = True
                    recognition_result = await recognize_image(file.path, user_question)
                    image_results.append(recognition_result)

                elif "pdf" in file.mime:
                    pdf_available = True
                    with pdfplumber.open(file.path) as pdf:
                        pdf_text = ''
                        for page in pdf.pages:
                            extracted_text = page.extract_text()
                            if extracted_text: 
                                pdf_text += extracted_text

                    if pdf_text: 
                        pdf_question.append(pdf_text)
                        # print(pdf_text)
    counter = cl.user_session.get("counter") + 1
    cl.user_session.set("counter", counter)
    try:
        response = None  
        answer =" "
        
        if not image_available and not pdf_available:
            write_request_data(user_question)
            response = await call_dialogflow(cl.user_session.get("session_id")) 
            await cl.sleep(2)
            if response and 'queryResult' in response:
                answer = response['queryResult']['responseMessages'][0]['text']['text'][0]
                last_response = cl.user_session.get("last_response")
                
                # print(chat_profile,last_response,counter)
                if (chat_profile=="Interview Room" and last_response is not None):  # Ensure there was a previous question from the model
                    qa_logger = cl.user_session.get('QA')  
                    qa_logger.add_question_answer(str(last_response), str(user_question))
                    cl.user_session.set("last_response", answer)
                    cl.user_session.set("QA",qa_logger)

                    if cl.user_session.get("counter") == 10:
                        # print("in the room")
                        CFID = cl.user_session.get("cfid")
                        cfobj =cl.user_session.get('CFOBJ')
                        probs = cl.user_session.get("probs")
                        submit_urls = cl.user_session.get("submit_urls")
                        cp_probs =  cl.user_session.get("cf_problems_list")
                        await cl.Message(content="Please solve this coding problem \n Problem 1",author ="Vertex AI").send()
                        res1 = await cl.AskActionMessage(
                            author = "Vertex AI",content= f"{probs[0]}\n Submit here: {submit_urls[0]}",
                            actions=[
                                cl.Action(name="Submit", value="Submit", label="✅ Continue"),
                                cl.Action(name="Can't Solve", value="Cancel", label="❌ Cancel"),
                            ],
                            timeout=1200
                        ).send()
                       
                        #Check if the response value is 'Submit' for problem 1

                        await cl.Message(content="Please solve this coding problem \n Problem 2",author= "Vertex AI").send()
                        res2 = await cl.AskActionMessage(
                            author = "Vertex AI",content=f"{probs[1]}\n Submit here :{submit_urls[1]}",
                            actions=[
                                cl.Action(name="Submit", value="Submit", label="✅ Continue"),
                                cl.Action(name="Can't Solve", value="Cancel", label="❌ Cancel"),
                            ],
                            timeout=1200
                        ).send()
                        
                        qa_logger = cl.user_session.get('QA')  
                        if(res1['value']=="Submit" and res2['value']=="Submit"):
                            
                            status =  cfobj.check_problems_solved(CFID,cp_probs)
                            for key,value in status.items():
                                qa_logger.add_question_answer(str(f"CODING PROBLEM{key}"), str(value))
                                await cl.Message(author = "Vertex AI",content = f"PROBLEM {key} : {value}").send()
                            # print(status)
                        cl.user_session.set("QA",qa_logger)        
                        
                        qa_logger = cl.user_session.get('QA') 
                        value = qa_logger.data
                        res = await evaluate_interview(value)
                        user_question = f'''
                        Thanks !! I would now like to end my interview.
                        I am providing you with my interview results please analyse them and present to me.
                        I want a full in depth analysis. Here is my interview review\n {res}
                        '''
                        write_request_data(user_question)
                        await call_dialogflow(cl.user_session.get("session_id")) 
                        answer = res
                        cl.user_session.set("last_response", answer)
            
            else:
                answer = "Sorry, I couldn't retrieve an answer."
        
        elif image_available and not pdf_available:
            await cl.sleep(2)
            answer = " ".join([str(result) for result in image_results])
        
        elif pdf_available and not image_available:
            user_question += "\n".join([str(question) for question in pdf_question])
            write_request_data(user_question)
            response = await call_dialogflow(cl.user_session.get("session_id"))  # Ensure this is awaited
            await cl.sleep(2)
            if response and 'queryResult' in response:
                answer = response['queryResult']['responseMessages'][0]['text']['text'][0]
            else:
                answer = "Sorry, I couldn't retrieve an answer."
        
        try:
            # answer =remove_tags(answer)
            answer_message = await cl.Message(content=answer , author ="Vertex AI").send()  
            if play_audios:
                audio_mime_type = "audio/mpeg"
                output_name, output_audio = await text_to_speech(answer, audio_mime_type)
                output_audio_el = cl.Audio(
                    name=output_name,
                    auto_play=True,
                    mime=audio_mime_type,
                    content=output_audio,
                )
                answer_message.elements = [output_audio_el]
                await answer_message.update()

        except Exception as e:
            await cl.Message(content=f"An error occurred: {str(e)}").send()

    except Exception as e:
        await cl.Message(
            content=f"An error occurred: {str(e)}"
        ).send()
        
@cl.on_message
async def main(message: cl.Message):  
    play_audios = cl.user_session.get("play_audios")

    image_results = []
    pdf_question= []
    image_available = False
    pdf_available = False
    files = message.elements  

    if files:  
        for file in files:
            if file.path is not None:
                if "image" in file.mime: 
                    image_available = True
                    recognition_result = await recognize_image(file.path, message.content)
                    image_results.append(recognition_result)

                elif "pdf" in file.mime:
                    pdf_available = True
                    with pdfplumber.open(file.path) as pdf:
                        pdf_text = ''
                        for page in pdf.pages:
                            extracted_text = page.extract_text()
                            if extracted_text: 
                                pdf_text += extracted_text

                    if pdf_text: 
                        pdf_question.append(pdf_text)
                       
    counter = cl.user_session.get("counter", 0) + 1
    cl.user_session.set("counter", counter)
    user_question = message.content
    chat_profile = cl.user_session.get("chat_profile")
    
    try:
        response = None  
        answer =" "
        if not image_available and not pdf_available:
            write_request_data(user_question)
            response = await call_dialogflow(cl.user_session.get("session_id")) 
            await cl.sleep(2)
            if response and 'queryResult' in response:
                answer = response['queryResult']['responseMessages'][0]['text']['text'][0]
                last_response = cl.user_session.get("last_response")

                if chat_profile=="Interview Room" and last_response:  # Ensure there was a previous question from the model
                    qa_logger = cl.user_session.get('QA')  
                    qa_logger.add_question_answer(str(last_response), str(user_question))
                    cl.user_session.set("last_response", answer)
                    cl.user_session.set("QA",qa_logger)

                    if cl.user_session.get("counter") == 10:
                        CFID = cl.user_session.get("cfid")
                        cfobj =cl.user_session.get('CFOBJ')
                        probs = cl.user_session.get("probs")
                        submit_urls = cl.user_session.get("submit_urls")
                        cp_probs =  cl.user_session.get("cf_problems_list")
                        await cl.Message(content="Please solve this coding problem \n Problem 1",author = "Vertex AI").send()
                        res1 = await cl.AskActionMessage(
                            author = "Vertex AI",content= f"{probs[0]}\n Submit here: {submit_urls[0]}",
                            actions=[
                                cl.Action(name="Submit", value="Submit", label="✅ Continue"),
                                cl.Action(name="Can't Solve", value="Cancel", label="❌ Cancel"),
                            ],
                            timeout=1200
                        ).send()
                    
                        await cl.Message(content="Please solve this coding problem \n Problem 2",author= "Vertex AI").send()
                        res2 = await cl.AskActionMessage(
                            author = "Vertex AI",content=f"{probs[1]}\n Submit here: {submit_urls[1]}",
                            actions=[
                                cl.Action(name="Submit", value="Submit", label="✅ Continue"),
                                cl.Action(name="Can't Solve", value="Cancel", label="❌ Cancel"),
                            ],
                            timeout=1200
                        ).send()
                        
                        qa_logger = cl.user_session.get('QA')  
                        if(res1['value']=="Submit" and res2['value']=="Submit"):
                            
                            status =  cfobj.check_problems_solved(CFID,cp_probs)
                            for key,value in status.items():
                                qa_logger.add_question_answer(str(f"CODING PROBLEM{key}"), str(value))
                                await cl.Message(author = "Vertex AI",content = f"PROBLEM {key} : {value}").send()
                                
                        cl.user_session.set("QA",qa_logger)        
                        
                        qa_logger = cl.user_session.get('QA') 
                        value = qa_logger.data
                        res = await evaluate_interview(value)
                        user_question = f'''
                        Thanks !! I would now like to end my interview.
                        I am providing you with my interview results please analyse them and present to me.
                        I want a full in depth analysis. Here is my interview review\n {res}
                        '''
                        write_request_data(user_question)
                        await call_dialogflow(cl.user_session.get("session_id")) 
                        answer = res
                        cl.user_session.set("last_response", answer)
            
            else:
                answer = "Sorry, I couldn't retrieve an answer."
        
        elif image_available and not pdf_available:
            await cl.sleep(2)
            answer = " ".join([str(result) for result in image_results])
        
        elif pdf_available and not image_available:
            user_question += "\n".join([str(question) for question in pdf_question])
            write_request_data(user_question)
            response = await call_dialogflow(cl.user_session.get("session_id"))  # Ensure this is awaited
            await cl.sleep(2)
            if response and 'queryResult' in response:
                answer = response['queryResult']['responseMessages'][0]['text']['text'][0]
            else:
                answer = "Sorry, I couldn't retrieve an answer."

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

    try:
        # answer =remove_tags(answer)
        answer_message = await cl.Message(content=answer , author ="Vertex AI").send()  
        if play_audios:
            audio_mime_type = "audio/mpeg"
            output_name, output_audio = await text_to_speech(answer, audio_mime_type)
            output_audio_el = cl.Audio(
                name=output_name,
                auto_play=True,
                mime=audio_mime_type,
                content=output_audio,
            )
            answer_message.elements = [output_audio_el]
            await answer_message.update()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()
