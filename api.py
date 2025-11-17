from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.ragcopy import user_call   
from fastapi.templating import Jinja2Templates

# Initialize FastAPI
app = FastAPI()

templates = Jinja2Templates(directory="static")
# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/")
def home():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    user_query = data.get("query", "")

    if not user_query:
        return {"answer": "Please enter a question."}

    answer = user_call(user_query)   
    return {"answer": answer}
