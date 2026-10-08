import re
def clean_text(t):
    t = str(t)
    t = re.sub(r"^.{0,120}?\(Reuters\)\s*-?\s*", "", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-zA-Z ]", " ", t).lower()
    t = re.sub(r"\breuters\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()