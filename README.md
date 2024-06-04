# SkillForge

## Introduction
The Challenge was to build a no/low-code conversational AI agent using Vertex AI Agent Builder that falls into one of four categories: Knowledge Bot, Lifestyle Bot, Productivity Booster and Customer-Facing Bot. Our idea falls into the Knowledge Bot Category
 where we build a Career Path Advisor that helps users navigate career paths by generating personalized skill development roadmaps, connecting them to relevant online courses and training programs, and offering interview practice with simulated questions and responses.

 ## Project Features
 1. ### Personalised Skill Development Roadmaps
    Our goal here is to offer users the capability to generate a personalized roadmap based on their current level of expertise and specific needs. To achieve this, we create an agent dedicated to roadmap generation.
    This agent utilizes a tool that leverages OpenAPI and Cloud Functions to search the web and compile resources, curating a customized roadmap tailored specifically for each user. We have incorporated a comprehensive prompt as well to achieve this.

 2. ### Recommendations for relevant online courses and certifications
    We create an agent that utilizes a specialized tool to search the web and compile a list of resources, including relevant online courses and certifications, tailored to the user's specific needs.
    This ensures that users receive personalized recommendations that align with their career goals and current skill levels.

 3. ### Stimulated Interview practice with feedback
    A key feature of our app is the interview simulation. We have created a dedicated agent specifically for this purpose. This agent will access a corpus of interview questions stored in a vector store to retrieve a list of relevant questions for the user. Using comprehensive prompts and examples in Vertex AI Agent Builder, the agent will simulate the interview experience by asking the user these questions and providing hints along the way.
    This interactive approach helps users prepare effectively for real interviews by offering a realistic practice environment.

 4. ### Coding Questions During Interview
    We utilize the CodeForces API to simulate the experience of solving coding questions during an interview. The agent will ask the user for their CodeForces ID and assign them two problems to solve on the CodeForces platform. Once the user indicates they have completed the problems, the agent will verify their completion status using the CodeForces API before proceeding further.
    This feature provides users with practical coding challenges, enhancing their problem-solving skills in a realistic interview setting.

 5. ### Networking tips and job search strategies
    Ye saundarya likhega

 6. ### CV Review
    We develop an additional agent specifically designed for CV review. Users can submit their CV to the agent, which then performs a thorough analysis using a detailed prompt.
    The agent will provide comprehensive feedback, highlighting strengths and areas for improvement, ensuring the user’s resume is polished and effective.

![TV - 1](https://github.com/himanshu-skid19/vertex-ai/assets/114365148/7cef9ca5-c4c4-4279-8a76-9cc618a30cd5)



## Installation and Setup    
### Setup 
1. Run `git clone https://github.com/himanshu-skid19/GoogleVertexAIHackathon.git`
2. Go to the directory by running `cd GoogleVertexAIHackathon`
3. Make sure to add your api keys in `config.py`
4. Create your `credentials.json` from google cloud auth and add it here.
5. Run `pip install -r requirements.txt`
6. Start the application by running `chainlit run app.py -w`

### Instructions to setup for windows

1. Install wsl
2. Open the project in wsl
3. Delete `libcuda.so` and `libcuda.so.1` from `C:\Windows\System32\lxss\lib` as administrator from windows/
4. In WSL run the following commands:
```
sudo ln -sr /mnt/c/Windows/System32/lxss/lib/libcuda.so.1.1 /mnt/c/Windows/System32/lxss/lib/libcuda.so.1
sudo ln -sr /mnt/c/Windows/System32/lxss/lib/libcuda.so.1.1 /mnt/c/Windows/System32/lxss/lib/libcuda.so
```

5. Now, to install kenlm, run the following commands:
```
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
```

6. Run ```pip install -r requirements.txt```.
## Demo Video
![demo vid](https://youtu.be/zrOSdnTL2gs)
## Project Architecture
```
├── README.md
├── app.py
├── audio_processing.py
├── chainlit.md
├── config.py
├── credentials.json
├── evaluate_interview.py
├── image_processing.py
├── request.json
├── requirements.txt
└── vertex_agent.py
```

## Team Members

1. Himanshu Singhal - [@himanshu-skid19](https://github.com/himanshu-skid19)
2. Sarthak Kapoor - [@sarthakkapoor44](https://github.com/sarthakkapoor44)
3. Saundarya Keshari - [@jazzsterq](https://github.com/jazzsterq)
4. Tanay Goenka - [@GreekGod01](https://github.com/GreekGod01)
