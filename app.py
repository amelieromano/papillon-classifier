import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(
    torch.load("papillon_model.pth", map_location="cpu", weights_only=True)
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── SVG illustrations (read once at startup) ──────────────────────────────────
with open("papillon_sitting.svg")  as f: SITTING_SVG  = f.read()
with open("papillon_running.svg")  as f: RUNNING_SVG  = f.read()
with open("papillon_portrait.svg") as f: PORTRAIT_SVG = f.read()

# ── Inference ─────────────────────────────────────────────────────────────────
def classify(image):
    if image is None:
        return ""

    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.nn.functional.softmax(output, dim=1)[0]
        papillon_pct     = probs[1].item() * 100
        not_papillon_pct = probs[0].item() * 100
        is_papillon      = papillon_pct > not_papillon_pct

    if is_papillon:
        return f"""
        <div class="result-container">
          <div class="dog-illustration">{RUNNING_SVG}</div>
          <div class="stamp-wrap">
            <div class="result-stamp positive-stamp">
              <span class="stamp-icon">🦋</span>
              <span class="stamp-text">PAPILLON!</span>
              <span class="stamp-conf">{papillon_pct:.1f}% confident</span>
            </div>
            <p class="result-note">yep — those ears are unmistakable ✨</p>
          </div>
        </div>"""
    else:
        return f"""
        <div class="result-container">
          <div class="dog-illustration">{SITTING_SVG}</div>
          <div class="stamp-wrap">
            <div class="result-stamp negative-stamp">
              <span class="stamp-icon">🐾</span>
              <span class="stamp-text">not a papillon</span>
              <span class="stamp-conf">{not_papillon_pct:.1f}% confident</span>
            </div>
            <p class="result-note">cute pup, but no butterfly ears here</p>
          </div>
        </div>"""

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Caveat:wght@400;700&family=Nunito:wght@400;600;700&display=swap');

/* === BASE === */
body, .gradio-container {
    background-color: #2C1A0E !important;
    font-family: 'Nunito', sans-serif !important;
}
.gradio-container {
    max-width: 880px !important;
    margin: 0 auto !important;
    padding-bottom: 3rem !important;
}
footer { display: none !important; }

/* grain overlay */
.gradio-container::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.72' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23g)'/%3E%3C/svg%3E");
    background-size: 180px 180px;
    opacity: 0.045;
    pointer-events: none;
    z-index: 9000;
    mix-blend-mode: overlay;
}

/* strip default Gradio panel chrome */
.block, .form, .panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* === HEADER === */
.zine-header {
    text-align: center;
    padding: 2.5rem 1rem 1.2rem;
}
.heading-wrap {
    display: inline-block;
    transform: rotate(-1deg);
}
.zine-heading {
    font-family: 'Fredoka One', cursive;
    font-size: 3.6rem;
    color: #F5EFE6;
    text-shadow: 5px 5px 0 #C4470A;
    line-height: 1;
    display: block;
    margin: 0;
}
.zine-subtitle {
    font-family: 'Caveat', cursive;
    font-size: 1.35rem;
    color: #C8882A;
    display: block;
    transform: rotate(1.5deg);
    margin-top: 0.3rem;
}
.hero-dog {
    margin: 1.4rem auto 0;
    display: flex;
    justify-content: center;
}
.hero-dog svg {
    filter: drop-shadow(3px 4px 0 #C4470A);
    width: 200px;
    height: auto;
}

/* === UPLOAD BOX === */
.upload-box {
    margin: 1.4rem 0 0.6rem;
    border: 3px solid #C4470A !important;
    box-shadow: 6px 6px 0 #C4470A !important;
    background: #1A0F07 !important;
    transform: rotate(-0.3deg);
    border-radius: 2px !important;
}
.upload-box label span {
    font-family: 'Caveat', cursive !important;
    font-size: 1.1rem !important;
    color: #C8882A !important;
}
/* dashed drop zone */
.upload-box .wrap {
    border: 2px dashed #C8882A !important;
    border-radius: 2px !important;
    background: transparent !important;
    color: #F5EFE6 !important;
}
.upload-box .wrap span {
    font-family: 'Caveat', cursive !important;
    color: #C8882A !important;
}

/* === BUTTON === */
#classify-btn {
    background: #C4470A !important;
    color: #F5EFE6 !important;
    font-family: 'Fredoka One', cursive !important;
    font-size: 1.35rem !important;
    letter-spacing: 0.04em !important;
    border: 3px solid #F5EFE6 !important;
    box-shadow: 5px 5px 0 #F5EFE6 !important;
    transform: rotate(0.5deg);
    padding: 0.55rem 2.2rem !important;
    border-radius: 2px !important;
    cursor: pointer;
    display: block;
    margin: 0.8rem auto !important;
    width: fit-content;
    transition: transform 0.08s, box-shadow 0.08s;
}
#classify-btn:hover {
    transform: rotate(0.5deg) translate(-2px, -2px) !important;
    box-shadow: 7px 7px 0 #F5EFE6 !important;
}
#classify-btn:active {
    transform: rotate(0.5deg) translate(3px, 3px) !important;
    box-shadow: 2px 2px 0 #F5EFE6 !important;
}

/* === RESULT AREA === */
.result-container {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    background: #1A0F07;
    border: 3px solid #C4470A;
    box-shadow: 7px 7px 0 #C4470A;
    transform: rotate(-0.4deg);
}
.dog-illustration {
    flex-shrink: 0;
}
.dog-illustration svg {
    width: 155px;
    height: auto;
    filter: drop-shadow(2px 3px 0 rgba(196,71,10,0.4));
}

/* === STAMP === */
.stamp-wrap { flex: 1; }

.result-stamp {
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0.7rem 1.3rem;
    border: 3px solid #F5EFE6;
    box-shadow: 5px 5px 0 #C4470A;
    transform: rotate(-1.5deg);
    margin-bottom: 0.9rem;
}
.positive-stamp { background: #C4470A; }
.negative-stamp {
    background: #2C1A0E;
    border-color: #C8882A;
    box-shadow: 5px 5px 0 #C8882A;
}

.stamp-icon { font-size: 1.9rem; line-height: 1; }
.stamp-text {
    font-family: 'Fredoka One', cursive;
    font-size: 1.9rem;
    color: #F5EFE6;
    text-shadow: 2px 2px 0 rgba(0,0,0,0.25);
    line-height: 1.15;
    text-transform: uppercase;
}
.negative-stamp .stamp-text { color: #C8882A; }

.stamp-conf {
    font-family: 'Caveat', cursive;
    font-size: 1.05rem;
    color: #F5EFE6;
    opacity: 0.82;
    margin-top: 0.15rem;
}
.result-note {
    font-family: 'Caveat', cursive;
    font-size: 1.2rem;
    color: #C8882A;
    margin: 0;
    transform: rotate(1deg);
    display: inline-block;
}

/* === FOOTER === */
.zine-footer {
    text-align: center;
    margin-top: 2.2rem;
    font-family: 'Caveat', cursive;
    color: #C8882A;
    font-size: 1rem;
    opacity: 0.65;
    transform: rotate(-0.4deg);
    display: block;
}
"""

# ── App ───────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:

    gr.HTML(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fredoka+One&family=Caveat:wght@400;700&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
    <div class="zine-header">
        <div class="heading-wrap">
            <span class="zine-heading">Is it a Papillon?</span>
            <span class="zine-subtitle">a tiny neural net for the world's most dramatic ears</span>
        </div>
        <div class="hero-dog">{PORTRAIT_SVG}</div>
    </div>
    """)

    image_input = gr.Image(
        label="drop a dog photo here",
        type="numpy",
        elem_classes=["upload-box"],
    )

    classify_btn = gr.Button("sniff it out →", elem_id="classify-btn")

    result_html = gr.HTML(value="")

    classify_btn.click(fn=classify, inputs=image_input, outputs=result_html)

    gr.HTML('<span class="zine-footer">ResNet18 · transfer learning · 63 training images · 91.7% accuracy</span>')

demo.launch()
