from Bob import Bob
import pandas as pd
from tqdm import tqdm
from icecream import ic

bob = Bob(promt_template="""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.  
Answer in German.
{context}
Question: {question}
Helpful Answer:""")

# bob.add_docs()

answers = {
    "Questions": [],
    "Temperatures": [],
    "Answers": [],
    "Sources": [],
    "Raw Result": []
}


def ask(temp, q):
    answers["Questions"].append(q)
    answers["Temperatures"].append(temp)
    result = bob.init_chain(temp, q)
    answers["Answers"].append(result["result"])
    answers["Sources"].append(result["source_documents"])
    answers["Raw Result"].append(result)


# Testing
with open("perfomance_questions.txt") as file_questions:
    questions = file_questions.readlines()
    for question in tqdm(questions, total=len(questions)):
        ic(f"Current question: {question}")
        ask(0, question)
        ask(0.1, question)
        ask(0.25, question)
        ask(0.5, question)

    df = pd.DataFrame(answers)
    df.to_csv("C:/Users/vruser01/OneDrive - P.ARC AG/100_Arbeit/perfomance_answers.csv")
