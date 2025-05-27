from typing import List
import argparse
import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


parser = argparse.ArgumentParser(description="Review generator argument parser")
parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate")
parser.add_argument("--output_file", type=str, default="result.csv", help="Output filename (.csv)")


llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
    model="gpt-4o",
)


class Review(BaseModel):
    review: str = Field(description="Generated review (follow the requirements)")
    label: int = Field(description="Label of generated review. 1 - if it is positive, 0 - otherwise")


class ReviewAnswer(BaseModel):
    reviews: List[Review]


PROMPT = """
## Your role
You are a data annotator and help enrich the dataset with new data.

## Your task
Your main goal is to generate {samples_num} samples for dataset in the following format:
{format_instructions}

## Dataset structure
Data is binary classification NLP problem typical set that contains
movie reviews. Some of them are negative, some are positive (0 or 1).

## Requirements
1. Generate unique reviews and original data only.
2. You can describe any film, actor, director, etc.
3. Your reviews must be shorter then 200 symbols.
4. Not necessary to name a certain movie, you can write abstract.
5. DO NOT REPEAT EXAMPLES!

## Examples

Positive review:
if you havent seen this movie than you need to It rocks and you have to watch it It is so funny and will make you laugh your guts out so you have to watch it and i saw it about a billion and a half times and still think it is funny so you have to yes i have memorized the whole movie and could quote it to you from start to finish you must see this move it is also cute because it is half a chick flick if you dont watch it then you are really missing outthis movie even has cute guys in it and that is always a bonus so in summary watch the movie now and trust me you will not be making a mistake did i mention the music is good too So you should like it if you enjoy music This is a movie that they rated correctly and it will work for anyone

Negative review:
This movie has no heart and no soul its an attempt to whomp up a cult film out of the leavings of other better directors principally David Lynch and Tim Burton Rifkin seems to think that if he overloads on a kind of rotted visual style and fills the street with crud and garbage hes making a statement But its not a statement ABOUT anything  except the directors shrill shriek of HEY LOOK AT ME IM AN ARTIST TOO But he doesnt have the imagination of an artist just a good memory for things that worked  such as some of the actors trapped in this  for other directors All of this would be almost acceptable if this movie was not a turgid boring chore to sit through

## Your answer here
"""


output_parser = JsonOutputParser(pydantic_object=ReviewAnswer)
chain = PromptTemplate.from_template(
    PROMPT,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
) | llm | output_parser


def generate(samples: int) -> ReviewAnswer:
    return ReviewAnswer.model_validate(chain.invoke({"samples_num": samples}))


if __name__ == "__main__":
    import pandas as pd

    args = parser.parse_args()
    samples = args.samples
    output_file = args.output_file

    response = generate(samples=samples)
    
    data = {"text": [], "label": []}
    for x in response.reviews:
        data["text"].append(x.review)
        data["label"].append(x.label)

    pd.DataFrame(data).to_csv(output_file, index=False)
