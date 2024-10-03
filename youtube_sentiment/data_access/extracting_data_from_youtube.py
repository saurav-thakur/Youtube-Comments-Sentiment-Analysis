import os
import pandas as pd
import googleapiclient.discovery
from dotenv import load_dotenv

load_dotenv()

def fetch_comments(videoId):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = os.environ.get("YOUTUBE_DEVELOPER_KEY")

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY
    )

    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=videoId,
            pageToken=next_page_token,
            maxResults=100  # Fetch up to 100 comments per page (max allowed by the API)
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                # comment['authorDisplayName'],
                # comment['publishedAt'],
                # comment['updatedAt'],
                # comment['likeCount'],
                comment['textDisplay']
            ])

        # Check if there is another page of comments
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    # Create DataFrame from the collected comments
    # df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
    df = pd.DataFrame(comments, columns=['text'])

    return df
