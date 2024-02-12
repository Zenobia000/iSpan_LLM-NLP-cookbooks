import requests
from pprint import pprint


def scrape_linkedin_profile():
    gist_response = requests.get(
        "https://gist.githubusercontent.com/Zenobia000/1ad3ffbb3ddcd8c496a1af8c0df5e900/raw/1a520b5df8650e10dfaaf809082cc8b3895e62ab/raydalio.json"
    )

    # print(f"this is {gist_response.json()['full_name']}'s linkedin")
    # pprint(gist_response.json())

    data = gist_response.json()

    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "None", None)
        and k not in ["people_also_viewed", "certifications"]
    }

    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
