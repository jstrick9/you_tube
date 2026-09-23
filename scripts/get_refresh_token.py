"""One-time helper: authorize AutoTube to upload to YOUR channel and print a refresh token.

Run this ONCE on your own computer (it opens a browser):

    pip install google-auth-oauthlib
    python scripts/get_refresh_token.py path/to/client_secret.json

Then store the printed values as GitHub secrets:
    YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN

IMPORTANT: In Google Cloud Console → Google Auth Platform → Audience, set the app's
publishing status to "In production" BEFORE running this. Tokens issued while the
app is in "Testing" expire after 7 days, which silently breaks automation.
"""
import json
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    secrets_file = sys.argv[1]
    flow = InstalledAppFlow.from_client_secrets_file(secrets_file, SCOPES)
    # When prompted, pick the Google account / Brand Account that OWNS the channel.
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
    info = json.load(open(secrets_file))
    client = info.get("installed") or info.get("web")
    print("\n✅ Success. Add these as GitHub repository secrets:\n")
    print(f"YT_CLIENT_ID      = {client['client_id']}")
    print(f"YT_CLIENT_SECRET  = {client['client_secret']}")
    print(f"YT_REFRESH_TOKEN  = {creds.refresh_token}\n")
    print("Keep these private. Never commit them to the repository.")


if __name__ == "__main__":
    main()
