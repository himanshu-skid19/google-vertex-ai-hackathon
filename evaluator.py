from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from config import claude_api_key
import os
import json

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