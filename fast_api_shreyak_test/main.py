from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {'message':'hi Shreyak, you have recieved canadian ita before 1/2/2024'}

@app.get("/square")
async def root():
    return {'message':'hi Shreyak, you have recieved canadian ita before 1/2/2024'}
