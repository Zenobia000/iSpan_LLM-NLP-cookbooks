import os
import requests
from dotenv import load_dotenv

load_dotenv()


def scrape_linkedin_profile(linkedin_profile_url: str):
    """
    scrape information from Linkedin profiles,
    manually scrape the information from the Linkedin profile
    :param linkedin_profile_url:
    :return: data for linked profile
    """

    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    headers = {"Authorization": "Bearer " + f'{os.environ.get("PROXYCRUL_API_KEY")}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=headers
    )

    # test_ans = f"this is {response.json()['full_name']}'s linkedin"

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
    }

    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


# test
# ans = scrape_linkedin_profile("https://www.linkedin.com/in/raydalio/")
