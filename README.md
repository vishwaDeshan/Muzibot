# Muzibot

## 1. Backend Setup (FastAPI)
```bash
cd backend
```

## 2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
```

## 3. Activate the virtual environment:

   - **Windows**:

   ```bash
   .\venv\Scripts\activate
   ```

   - **Linux/Mac**:

   ```bash
   source venv/bin/activate
   ```
## 4. Instal required packages
```bash
pip install -r requirements.txt
```

## 5. Run the FastAPI backend:

```bash
uvicorn app.main:app --reload
```
