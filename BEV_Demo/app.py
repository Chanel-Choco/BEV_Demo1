"""
Streamlit proof-of-concept: upload an artwork -> P(AI-generated) from Model A, Model B and their ensemble,
plus optional Grad-CAM. Run with:   streamlit run app.py
"""
import time

import streamlit as st

from inference import DEVICE, gradcam_overlay, load_models, predict_prob, prepare_image

st.set_page_config(page_title="AI Artwork Detector (thesis demo)", page_icon="🎨", layout="wide")


@st.cache_resource(show_spinner="Loading model weights...")
def get_models(weights_dir):
    return load_models(weights_dir)


st.title("🎨 AI-generated artwork detector")
st.caption("Thesis proof of concept: lightweight CNNs (MobileNetV3-Small / EfficientNet-B0) with a Learnable "
           "Frequency Attention Block (LFAB) and Grad-CAM explanations.")

with st.sidebar:
    st.header("Settings")
    weights_dir = st.text_input("Weights folder", value="weights",
                                help="Folder containing mobilenet_lfab_model.pth and efficientnet_lfab_model.pth")
    threshold = st.slider("Decision threshold (P(AI) ≥ this → 'AI-generated')", 0.30, 0.90, 0.50, 0.01,
                          help="0.5 is what the notebooks report as the default.")
    show_cam = st.checkbox("Show Grad-CAM heatmaps", value=True)
    st.caption(f"Running on: **{DEVICE}**")

models, missing = get_models(weights_dir)
if missing:
    st.error("Some checkpoints were not found:\n\n" + "\n".join(f"- {m}" for m in missing))
if not models:
    st.info("Copy `mobilenet_lfab_model.pth` (Notebook 2 output) and `efficientnet_lfab_model.pth` "
            "(Notebook 3 output) into the weights folder, then refresh.")
    st.stop()

up = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png", "webp"])

with st.expander("About this demo / limits"):
    st.markdown(
        "- Trained on 224×224 crops of artwork from ArtBench, WikiArt, MidJourney and Stable Diffusion. "
        "In-distribution test accuracy is roughly 90–93%.\n"
        "- On two generators never seen in training, accuracy drops to roughly 82–84%.\n"
        "- On photo-style images (CIFAKE) it is close to chance, so don't feed it photographs.\n"
        "- The score is a model probability, not proof. Heavy compression, noise or resizing lower reliability."
    )

if up is None:
    st.stop()

# ---- step-by-step run, so the audience can see what is happening -----------------------------
results, cams = {}, {}
with st.status("Analysing image...", expanded=True) as status:
    st.write("1/4  Preprocessing (RGB → 256×256 → center-crop 224 → normalise), same as training")
    img256, tensor = prepare_image(up)
    time.sleep(0.2)

    st.write("2/4  Running the models")
    for label, model in models.items():
        t0 = time.perf_counter()
        results[label] = (predict_prob(model, tensor), (time.perf_counter() - t0) * 1000)

    st.write("3/4  Combining (ensemble = average of the models' probabilities)")
    ensemble = sum(p for p, _ in results.values()) / len(results)

    if show_cam:
        st.write("4/4  Grad-CAM: where did each model look?")
        for label, model in models.items():
            cams[label] = gradcam_overlay(model, tensor)
    status.update(label="Done", state="complete", expanded=False)

# ---- results ---------------------------------------------------------------------------------
verdict_ai = ensemble >= threshold if len(models) > 1 else next(iter(results.values()))[0] >= threshold
left, right = st.columns([1, 1.4])
with left:
    st.image(img256.crop((16, 16, 240, 240)), caption="What the models see (224×224 center crop)")
with right:
    st.subheader("🤖 Likely AI-generated" if verdict_ai else "🖌️ Likely human-made")
    st.progress(min(max(ensemble, 0.0), 1.0), text=f"Ensemble P(AI) = {ensemble:.1%}   (threshold {threshold:.0%})")
    for label, (p, ms) in results.items():
        st.write(f"**{label}**: P(AI) = {p:.1%}  ·  {ms:.0f} ms")
    if abs(ensemble - threshold) < 0.10:
        st.warning("Close to the threshold: treat this as uncertain.")

if cams:
    st.subheader("Grad-CAM (red = regions pushing the score toward 'AI-generated')")
    cols = st.columns(len(cams))
    for col, (label, overlay) in zip(cols, cams.items()):
        col.image(overlay, caption=label)
