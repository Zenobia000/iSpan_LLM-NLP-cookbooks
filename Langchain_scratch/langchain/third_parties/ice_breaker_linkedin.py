from linkedin import scrape_linkedin_profile
# from linkedin_request_github_gist import scrape_linkedin_profile

from agents import linkedin_lookup_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("hello langchain")

    linkedin_profile_url = linkedin_lookup_agent.lookup(name = "elon mask")

    summary_template = """
    given the information {information} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # linkedin_profile_url = "https://www.linkedin.com/in/raydalio/"
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url = linkedin_profile_url
    )

    print(chain.run(information=linkedin_data))



