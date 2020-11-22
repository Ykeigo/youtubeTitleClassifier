from apiclient.discovery import build

YOUTUBE_API_KEY = 'AIzaSyBIVx2mBxOiqNZnbrnnxp8vltWf_Yf_glM'

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

search_response = youtube.search().list(
part='snippet',
#検索したい文字列を指定
q='ボードゲーム',
#視聴回数が多い順に取得
order='viewCount',
type='video',
).execute()

print(len(search_response['items']))