# Setup Guide — From Zero to Deployed

## What you need
- A Google account (Gmail) — that's it for the API key
- Python 3.10+ on your machine
- Git on your machine
- A GitHub account → https://github.com
- A Railway account → https://railway.app (sign in with GitHub, free)

---

## Step 1 — Get your free Gemini API key (2 min, no credit card)

1. Go to → https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click **Create API key**
4. Click **Create API key in new project**
5. Copy the key — looks like `AIzaSy...`

That's it. No billing, no credit card, free forever up to 1,000 requests/day.

---

## Step 2 — Run locally to verify it works (5 min)

```bash
# go into the project folder
cd relevant-priors

# install dependencies
pip install -r requirements.txt

# set your API key (Mac/Linux)
export GEMINI_API_KEY="AIzaSy..."

# set your API key (Windows)
set GEMINI_API_KEY=AIzaSy...

# start the server
python main.py
# → running at http://localhost:8000
```

Open a second terminal and test it:
```bash
python test.py
```

You should see predictions with an accuracy score. If you see errors, double
check that the API key is set and the server is still running.

---

## Step 3 — Push to GitHub (3 min)

```bash
git init
git add .
git commit -m "relevant priors api"
```

Go to https://github.com/new
- Repository name: `relevant-priors`
- Keep everything default (public or private, both work)
- Click **Create repository**

Run the two commands GitHub shows you:
```bash
git remote add origin https://github.com/YOUR_USERNAME/relevant-priors.git
git branch -M main
git push -u origin main
```

---

## Step 4 — Deploy on Railway (5 min)

1. Go to https://railway.app
2. Click **Login** → **Login with GitHub**
3. Click **New Project**
4. Click **Deploy from GitHub repo**
5. Pick `relevant-priors` from the list
6. Click **Deploy Now**

Railway detects Python automatically and starts building. Takes ~2 minutes.
Watch the logs — you'll see it install dependencies and start the server.

---

## Step 5 — Add the API key on Railway

The server will crash without it. Add it here:

1. Click on your service (the card in the Railway dashboard)
2. Click the **Variables** tab
3. Click **New Variable**
4. Name: `GEMINI_API_KEY`
5. Value: your key starting with `AIzaSy...`
6. Click **Add**

Railway redeploys automatically. Takes about 1 minute.

---

## Step 6 — Get your public URL

1. Click your service → **Settings** tab
2. Scroll to **Networking**
3. Click **Generate Domain**
4. You'll get something like:
   `https://relevant-priors-production.up.railway.app`

Verify it's alive — open this in your browser:
```
https://relevant-priors-production.up.railway.app/health
```
Should show: `{"status":"ok","cache_size":0}`

Test the full flow:
```bash
python test.py --url https://relevant-priors-production.up.railway.app/predict
```

---

## Step 7 — Submit

Three things needed:

1. **Endpoint URL**
   ```
   https://relevant-priors-production.up.railway.app/predict
   ```

2. **Code zip** — zip the project folder and upload

3. **Write-up** — use `WRITEUP.md`

---

## Troubleshooting

**"couldn't connect" on test.py (local)**
→ Make sure `python main.py` is running in a separate terminal

**Railway build fails**
→ Go to Deployments tab → click the failed deploy → read the logs
→ Usually a package version issue — try removing version numbers from requirements.txt

**Getting 500 errors on Railway**
→ Check Railway logs
→ 99% of the time it's the GEMINI_API_KEY variable missing or wrong

**Predictions are all true**
→ The server hit a parse error and defaulted to true (safe fallback)
→ Check Railway logs for "couldn't parse response"

**Want to redeploy after a code change**
```bash
git add .
git commit -m "fix"
git push
```
Railway auto-deploys on every push.
