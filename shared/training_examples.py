
"""
Training examples used by the lightweight depression severity model.
"""

import os
import re
import textwrap
from typing import Optional, Tuple

_EXPLANATION_NEWLINES = re.compile(r"\n{3,}")

_PARAGRAPH_KEY_ORDER = (
    "id",
    "severity",
    "text",
    "prediction_confidence",
    "prediction_label",
    "SHAP_explanation",
    "RAG_explanation",
    "COUNTERFACTUAL_explanation",
    "HYBRID_explanation",
)

_EXPLANATION_KEY_MAP = {
    key.lower(): key for key in _PARAGRAPH_KEY_ORDER if key.endswith("_explanation")
}


def _normalize_explanation(explanation: str) -> str:
    if not explanation:
        return ""
    text = explanation.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = _EXPLANATION_NEWLINES.sub("\n\n", text)
    return text


def _find_example_index(paragraph_id: str) -> Optional[int]:
    if not paragraph_id:
        return None
    for i, row in enumerate(PARAGRAPHS):
        if row.get("id") == paragraph_id:
            return i
    return None


def get_cached_prediction(paragraph_id: str) -> Tuple[Optional[str], Optional[float]]:
    idx = _find_example_index(paragraph_id)
    if idx is None:
        return None, None
    row = PARAGRAPHS[idx]
    return row.get("prediction_label"), row.get("prediction_confidence")


def get_cached_explanation(paragraph_id: str, use_case_name: str) -> Optional[str]:
    idx = _find_example_index(paragraph_id)
    if idx is None:
        return None
    row = PARAGRAPHS[idx]
    key = f"{use_case_name}_explanation"
    if key in row:
        return _normalize_explanation(row.get(key) or "") or None
    arch = row.get("selected_architecture") or {}
    entry = arch.get(use_case_name) or {}
    return _normalize_explanation(entry.get("explanation_text") or "") or None


def save_prediction(paragraph_id: str, label: str, confidence: float) -> None:
    idx = _find_example_index(paragraph_id)
    if idx is None:
        return
    PARAGRAPHS[idx]["prediction_label"] = label
    PARAGRAPHS[idx]["prediction_confidence"] = confidence
    _write_paragraphs()


def save_explanation(paragraph_id: str, use_case_name: str, explanation: str) -> None:
    idx = _find_example_index(paragraph_id)
    if idx is None:
        return
    row = PARAGRAPHS[idx]
    row[f"{use_case_name}_explanation"] = _normalize_explanation(explanation)
    _write_paragraphs()


def _format_string_value(value: str, indent: int) -> str:
    """
    Serialize a string using a raw triple-quoted string so that:
    - Newlines (\\n) are stored as real newlines, preserving paragraph breaks.
    - Bold markers (**text**) are never split across lines by pprint.
    - The closing triple-quote sits on its own line at the correct indent level.
    """
    escaped = value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    inner_indent = " " * (indent + 4)
    close_indent = " " * indent
    return f'(\n{inner_indent}"""\\\n{inner_indent}{escaped}\n{close_indent}""")'


def _format_value(value, indent: int) -> str:
    """Recursively format a dict value for writing back to the source file."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # Use triple-quoted block only when the string contains newlines or **markers**
        if "\n" in value or "**" in value or len(value) > 80:
            return _format_string_value(value, indent)
        return repr(value)
    if isinstance(value, dict):
        return _format_dict(value, indent)
    if isinstance(value, list):
        return _format_list(value, indent)
    return repr(value)


def _format_dict(d: dict, indent: int) -> str:
    inner = " " * (indent + 4)
    close = " " * indent
    lines = []
    for k, v in d.items():
        formatted_v = _format_value(v, indent + 4)
        lines.append(f"{inner}{repr(k)}: {formatted_v},")
    body = "\n".join(lines)
    return f"{{\n{body}\n{close}}}"


def _format_list(lst: list, indent: int) -> str:
    inner = " " * (indent + 4)
    close = " " * indent
    lines = []
    for item in lst:
        formatted = _format_value(item, indent + 4)
        lines.append(f"{inner}{formatted},")
    body = "\n".join(lines)
    return f"[\n{body}\n{close}]"


def _normalize_paragraph_entry(row: dict) -> dict:
    remapped = {}
    for key, value in row.items():
        mapped_key = _EXPLANATION_KEY_MAP.get(key.lower(), key)
        if mapped_key in remapped:
            continue
        if mapped_key.endswith("_explanation") and isinstance(value, str):
            remapped[mapped_key] = _normalize_explanation(value)
        else:
            remapped[mapped_key] = value

    ordered = {}
    for key in _PARAGRAPH_KEY_ORDER:
        if key in remapped:
            ordered[key] = remapped[key]
    for key, value in remapped.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def normalize_paragraphs() -> None:
    """Rewrite PARAGRAPHS on disk using the standard key order."""
    _write_paragraphs()


def _write_paragraphs() -> None:
    file_path = os.path.abspath(__file__)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    normalized = [_normalize_paragraph_entry(row) for row in PARAGRAPHS]

    # Build the PARAGRAPHS block using our custom formatter so that
    # multi-line strings and **bold** markers are never mangled.
    new_block = "PARAGRAPHS = " + _format_list(normalized, indent=0)

    # Replace the old PARAGRAPHS assignment (handles the full block to EOF safely).
    content = re.sub(
        r"PARAGRAPHS\s*=\s*\[.*\]\s*$",
        new_block,
        content,
        flags=re.DOTALL,
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)



PARAGRAPHS = [
    {
        'id': 'daic_woz_severe_321',
        'severity': 'severe',
        'text': (
            """\
                                                            Lately, I’ve been feeling deeply depressed and weighed down by worries. My life feels heavy; I had to settle for a retail job because I couldn't find work in my field, and my daughter is currently battling cancer, leaving me frustrated and anxious. I haven't had a good night's sleep in a year, waking up every few hours. This severe fatigue leaves me groggy, lacking energy, and struggling to concentrate, causing me to forget things and make mistakes at work. I’ve experienced a painful shift in my social functioning and interests; I haven't gone out to dinner with a close friend in over a year. My self-worth is low—I don't know my best qualities and regret my lack of education— and my appetite is disrupted.Though I find moments of pure joy playing with my granddaughter, I mostly feel trapped by sadness and failure, coping by keeping quiet when struggling.
        
        
        
        
        """),
        'prediction_confidence': 0.709019487708329,
        'prediction_label': 'severe',
        'SHAP_explanation': (
            """\
            Based on what you shared, the model flagged a **severe** level of depression.  
The language you used, such as **quiet**, hints at holding emotions inside, which the system associates with isolation. The mention of **sadness** signals a persistent negative mood, a strong indicator of deeper distress. Saying your **daughter** is battling cancer adds a caregiving burden that strongly correlates with higher depressive risk.

These particular words helped the tool weigh the overall picture, highlighting the intensity of the struggle you’re experiencing. For example
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Your screening result is **severe**.

The words that pushed the model toward the severe label were **quiet**, **sadness**, and **trapped**.  Saying you feel quiet and pull back from talking shows isolation.  Describing deep sadness points to a lasting low mood, and feeling trapped signals that the situation feels unchangeable.  Those three words together paint a picture of significant emotional weight and difficulty moving forward, which the system interprets as a higher level of distress.

If the message had said, *“I still keep in touch with friends and feel hopeful sometimes”* instead of *“I’ve been quiet and left everything behind,”* the assessment might have shifted toward a moderate level.
        """),
    },
    {
        'id': 'daic_woz_severe_346',
        'severity': 'severe',
        'text': (
            """\
                                                            Lately, I’ve been feeling really sad, stressed, and miserable, carrying a severe heaviness from my past hardships and trauma. Sleeping is incredibly difficult; I lie awake dwelling on stressors or wake up shaking from intense nighttime panic attacks and vivid nightmares. This constant insomnia leaves me completely exhausted, moody, and struggling with severe fatigue, poor concentration, and a disrupted appetite. My self-worth is low—I often forget my good qualities, regret past choices like my last marriage, and feel a deep sense of failure. Socially, my family can be judgmental, and though I have a loving boyfriend, I struggle with an internal urge to withdraw. I have to forcefully push myself to attend auditions or make it to appointments because a part of me just wants to hide away. While I try to survive and stay busy to avoid feeling that life isn't worth it, everything feels like an agonizing struggle.
        
        
        
        
        """),
        'prediction_confidence': 0.7591856923885394,
        'prediction_label': 'severe',
        'SHAP_explanation': (
            """\
            Your screening shows a **severe** level of depressive symptomatology.  
The system highlighted a few strong signals such as **nighttime panic attacks**, **low self-worth**, and the repeated sense of **exhaustion**. These phrases suggest intense, ongoing pain and a deep‑seated belief that things feel overwhelming or unmanageable, which the model uses to gauge the overall severity. The presence of these markers indicates a high level of distress that the algorithm identifies as typical of more serious depression.
        """),
        'RAG_explanation': (
            """\
            Your assessment indicates a *severe* level.  
**Lately, I’ve been feeling really sad, stressed, and miserable** points to the kind of ongoing sadness that feels heavy most of the day and pulls you into a low mood.

The second clue is **I often forget my good qualities, regret past choices ... and feel a deep sense of failure**—this shows you think you don’t deserve happiness and are very harsh on yourself.  
The final hint comes from **I have to forcefully push myself to attend auditions ... because a part of me just wants to hide away**, which highlights that you’re no longer enjoying activities or interacting with others the way you used to.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Your message was assessed as **severe**.  
The AI put a lot of weight on a few striking phrases.  For instance, **“sleeping is incredibly difficult”** signals a strong sleep problem, a common sign that feelings are deep and persistent.  **“Self‑worth is low”** shows a persistent negative self‑image that the model sees as a major warning.  Phrases that highlight regret, such as **“regret past choices like my last marriage,”** give further evidence of lasting emotional distress.

A simple change that might soften the reading would be: replace “I wake up shaking from intense panic attacks and vivid nightmares” with “I wake up feeling somewhat rested.” A calmer description of sleep would lower the severity in the model’s view.
        """),
        'HYBRID_explanation': (
            """\
            The model is saying your depression level is **SEVERE**.  
This comes mainly from how you describe feeling **low self‑worth**, having persistent **insomnia**, and wanting to withdraw from people (even when you have supportive relationships). These themes point to deep sadness that lasts most days, a sense of worthlessness, and a continual struggle to function, all of which push the estimate higher.

For example, if the message said “I get good sleep” instead of “I lie awake and panic,” the interpretation would lean toward a lower severity assessment.
        """),
    },
    {
        'id': 'daic_woz_severe_348',
        'severity': 'severe',
        'text': (
            """\
                                                            Lately, I’ve been feeling so tired, sad, depressed, and blue. I was diagnosed with depression about a year ago because I just couldn't pull myself out of it, and right now, I’m simply not happy. My sleep is terrible—I barely sleep at all, which leaves me feeling crazy, distracted, and struggling to cope with even mundane daily things. This severe lack of motivation, energy, and low self-worth makes me feel like I'm not my usual self. Even my appetite has been completely disrupted. To cope, I’ve started staying to myself and withdrawing, not going out with friends like I used to. My new relationship has lost its spark and just feels okay now, and the cold weather easily brings me down. Although going to therapy helps me get things off my chest, and I still feel proud and fulfilled by my children, everyday life currently feels like an overwhelming weight.
        
        
        
        
        """),
        'prediction_confidence': 0.6679306370944829,
        'prediction_label': 'severe',
        'RAG_explanation': (
            """\
            The model indicates a **severe** level of depression.  
You describe feeling **so tired, sad, depressed, and blue**, which shows a persistent low mood that lingers for most of the day.  When you say you’re **barely sleeping at all**, it reflects the constant fatigue that saps energy from everyday activities.  Your mention of **low self‑worth** points to ongoing feelings of shame and unworthiness that push you away from people and responsibilities.

This combination of symptoms—ongoing sadness, relentless tiredness, and deep self‑criticism—matches the pattern seen in severe depression. The fact that these issues affect almost every day and interfere with work, relationships, and your sense of self makes the overall assessment gravitate toward a severe classification.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            I’m saying your check scored **SEVERE**. That means the words you used show a lot of deep sadness and trouble. For example, when you wrote you were **blue**, the model saw a very strong feeling of low mood. The word **right** often signals that this is how you actually feel, not just a metaphor. Lastly, “**cope**” signals you’re struggling to handle day‑to‑day life, which adds to the heaviness in the score.

One quick test is: if the text had said you were **feeling hopeful** instead of **blue**, the result could have been lighter, possibly moving toward a lower level. It shows how one word can change the whole picture.
        """),
        'HYBRID_explanation': (
            """\
            The assessment says you’re experiencing **SEVERE** depression.  
Words like **blue**, **barely**, and **right** stood out because they hint at deep sadness, very little sleep, and a sense that everyday life feels heavy—a pattern that signals the most intense level of distress.

If the wording had said “I’m **happy** instead of **blue**,” it would suggest a lighter mood, which might move the rating toward a milder level.
        """),
    },
    {
        'id': 'daic_woz_severe_362',
        'severity': 'severe',
        'text': (
            """\
                                                            Lately, I’ve been feeling so-so, run down, and a bit stressed, struggling with ongoing depression, lack of interest, and too many regrets weighing on my mind. Sleep is never easy and always bad, leaving me tired, lethargic, and without energy. It makes it incredibly hard to keep my thoughts in order or manage the absolute basics of my day. My appetite has also been severely disrupted. Because of my past trauma from a near-fatal stalker attack, I find myself battling complex cognitive and emotional patterns. To cope, I actively use the proactive tools from therapy and reach out to others rather than isolating myself. While I try to stay rational, calm, and reliable, I can't even remember the last time I felt genuinely happy. Despite these severe struggles with self-worth and exhaustion, my children remain my greatest pride, and working with animals still brings a brief smile to my face.
        
        
        
        
        """),
        'prediction_confidence': 0.7229865989314259,
        'prediction_label': 'severe',
        'SHAP_explanation': (
            """\
            Your screening indicates a **SEVERE** level of depression.

The word **exhaustion** shows how much energy you’re losing, making daily tasks feel almost impossible. The phrase **lack of interest** points to anhedonia—the loss of pleasure in activities you used to enjoy. When you say you’ve never felt genuinely happy **last time**, it signals that even brief moments of joy are missing.

These elements together paint a picture of a deeply intense depressive experience, helping explain why the assessment flagged a severe risk.
        """),
        'RAG_explanation': (
            """\
            The system says the picture of how you’re feeling is **severe**.

When you say you’re **“feeling so-so, run down, and a bit stressed”**, it shows the kind of everyday sadness and hopelessness that can make even simple tasks feel heavy. Your words about a **“lack of interest”** and not being able to remember the last time you felt genuinely happy point to a loss of pleasure in the things you once enjoyed. The sense that you’re “not good enough” and that things feel pointless mirrors the feeling of being worthless that many people experience when they’re struggling deeply.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Your assessment points to a **severe** level of depression.  
The model focused on parts of your message that describe intense fatigue and persistent negative feelings. For example, the phrase **“exhaustion”** signals ongoing tiredness and a lack of energy, while **“children remain my greatest pride”** shows a positive anchor that somewhat balances the score but doesn’t bring it down because the rest of the text tells a more painful story. These mixed signals help the system decide that the overall mood is still quite low.

A small shift could tilt the judgment.  
If the message said **“I feel rested when I sleep”** instead of **“sleep is never easy”**, the positive sleep experience would likely move the result from severe toward moderate.
        """),
        'HYBRID_explanation': (
            """\
            Your reflection was judged as showing **severe** depressive risk. Words like **lack of interest**, **ongoing depression**, and **exhaustion** point to a pattern of feeling stuck, sad, and drained every day. **Lack of interest** means activities no longer bring joy; **ongoing depression** indicates sadness that sticks around almost all the time; **exhaustion** shows a persistent lack of energy that makes even small tasks feel heavy. These clues match core symptoms clinicians look for when assessing how hard it can be to get through a morning or find pleasure in familiar activities.  

If the message had said something like “I am feeling happy and full of energy” instead of the current wording, the assessment would likely shift toward a lower level of concern. Showing sustained positivity often signals a milder mood state.
        """),
    },
    {
        'id': 'daic_woz_severe_367',
        'severity': 'severe',
        'text': (
            """\
                                                            Lately, I feel pensive and down, carrying a heavy sense of hopelessness, severe depression, and a total loss of interest in life. Since losing my job and moving to LA, I feel like a complete failure who can't get back on my feet, leading to estrangement from my family and friends. I constantly dwell on past mistakes, especially a failed relationship that spiraled out of control, directing my anger inward with intense guilt. My mind is always overwhelmed; everything triggers painful memories, forcing me to double-task to keep my brain occupied, though concentrating remains extremely difficult. Physically, I am plagued by light, restless sleep, low energy, and appetite issues. To cope with this emotional instability and stay sober, I attend AA meetings to put my formless pain into words. I wonder if I'll ever feel like a normal, happy person again, remembering a Christmas years ago when life felt whole.
        
        
        
        
        """),
        'prediction_confidence': 0.7093200663893708,
        'prediction_label': 'severe',
        'COUNTERFACTUAL_explanation': (
            """\
            Your screening result shows **SEVERE** depression. The assessment was pulled mainly from phrases like **severe depression**, **hopelessness**, and **estrangement**. These words signal a strong sense of despair, loss of motivation, and social isolation—symptoms that the model treats as high‑level indicators of deep emotional distress. The presence of these terms together increases the overall signal that the person is experiencing intense, persistent difficulty. These words align with clinical markers of mood disorders, showing a sustained negative outlook and isolation that clinicians flag when diagnosing.

If, for example, the message said **“I feel hopeful about the future”** instead of **“I feel hopeless,”** the model would see a less intense expression of despair, which could lower the predicted severity to a more moderate level.
        """),
        'HYBRID_explanation': (
            """\
            Overall, your screening shows a severe level of depression overall severity right now. The AI looks for clues that match everyday expressions of feeling overwhelmed. In your text, phrases like **failure** and **estrangement** stand out because they signal deep self‑criticism and broken connections, which the model treats as strong indicators of major depressive symptoms. Even though you also used words like *feel* and *family*, which help lower the odds, the negative signals were stronger and pushed the result toward severe.

Additionally, if the text had said **success** instead of **failure**, the message would have shown more confidence, and the prediction could have shifted toward moderate or mild instead of severe today.
        """),
    },
    {
        'id': 'daic_woz_severe_426',
        'severity': 'severe',
        'text': (
            """\
                                                            Lately, I’ve been feeling not good at all, carrying a deep sense of depression and hopelessness. Since being released from prison, I feel older, overwhelmed by responsibilities, and like my life isn't where it's supposed to be, which leaves me struggling with self-worth and feeling sorry for myself. Sleep is never easy, and I am plagued by severe restlessness, severe low energy, and appetite disruption. My concentration is impaired, making it hard to focus, yet I remain fully determined not to give up. Because of past traumas and my PTSD diagnosis, I sometimes experience intense emotional instability and flashbacks of past violence, fights, and shootouts that inhibit me. To cope, I actively rely on therapy, which teaches me to step back and rationally assess situations instead of reacting purely on emotion. Although I still enjoy music and social gatherings, true happiness feels far away, anchored in memories of mutual love.
        
        
        
        
        """),
        'prediction_confidence': 0.7380475438759672,
        'prediction_label': 'severe',
        'SHAP_explanation': (
            """\
            Your assessment indicates a **Severe** level of depression.  
This means the signals in your writing point to a strong presence of distress.

Key phrases that the model highlighted were **traumas**, **plagued**, and **flashbacks of past violence**. These words emphasize deep, ongoing memories and sensations that can keep your mood low and your concentration scattered, which are typical markers of severe depressive patterns. Other parts of your text, like saying you **feel** or **rely** on therapy, were seen as less concerning and actually lowered the overall signal.
        """),
        'RAG_explanation': (
            """\
            Your test indicates a **severe** level of symptom intensity. In your own words, you say you’ve been feeling **“deep sense of depression and hopelessness”** almost every day, that you’re **“feeling older, overwhelmed by responsibilities”**, and that **“my concentration is impaired”**. These statements suggest you’re experiencing a persistent down mood, a strong sense of worthlessness or failure, and an ongoing lack of mental clarity. Together, they paint a picture of intense sadness that shows up throughout the day, a belief that you’re not meeting expectations, and a steady difficulty keeping your mind on tasks, all of which align with the severe category.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Your screening result is **severe**.

The system looked most closely at phrases like **“traumas”**, **“plagued”**, and **“emotional instability”** (including the flashbacks you mention). These words signal chronic distress, ongoing symptoms, and emotional upheaval that are common in more intense cases. They give the assessment a sense that the overall picture is heavy and difficult to manage.

If the message had said something like *“I’ve been sleeping well and have plenty of energy”* instead of *“Sleep is never easy, and I am plagued by severe restlessness”*, the focus would shift away from those markers of extreme difficulty, which could pull the result toward a lighter category.
        """),
        'HYBRID_explanation': (
            """\
            The result is **severe**, meaning the wording shows a strong signal of deep, ongoing depression.

Key parts that pushed it toward that level are the words **“traumas”**, **“plagued”**, and **“focused”**. These phrases signal intense emotional pain, constant distress, and difficulties staying on task—each is a core sign of serious depressive symptoms. The mention of PTSD and flashbacks also adds to the intensity, showing how past hurt is still affecting you daily.

If the message said “I feel fine” instead of “I feel not good at all,” the overall tone would shift toward a lighter mood and the result might become moderate or mild. The change in wording would reduce the intensity of the depressive indicators.
        """),
    },
    {
        'id': 'daic_woz_severe_440',
        'severity': 'severe',
        'text': (
            """\
                                                            Lately, I’ve been feeling very stressed, moody, and irritable, with my mind constantly jumping from one thought to another, making concentration incredibly difficult. My son’s incarceration has been a devastating stressor, leaving me feeling deeply depressed and hopeless. Between worrying about my children and having racing thoughts, getting a good night’s sleep is nearly impossible, leaving me completely exhausted with very low energy. My appetite is significantly disrupted, and I struggle with feelings of failure and self-worth regarding marital arguments and parenting mistakes. Despite being naturally outgoing, I tend to withdraw and avoid talking entirely when upset, hiding my emotions. To cope and relax, I design jewelry. Though I heavily relied on therapy, losing my therapist due to funding cuts has left me to navigate this emotional instability completely alone. I try to remain a compassionate go-getter for my family, but right now, finding genuine happiness is a real struggle.
        
        
        
        
        """),
    },
    {
        'id': 'daic_woz_moderate_319',
        'severity': 'moderate',
        'text': (
            """\
                                                            Lately, I’ve been feeling down and not like myself, struggling with a persistent sense of lethargy and low energy that often leaves me lying around. Since being diagnosed with depression a year ago, I face a regular loss of interest in activities, though watching USC football can still lift my spirits. My sleep is frequently disrupted, leaving me irritable and cranky, and I notice regular appetite changes. Almost daily, I face severe concentration difficulties. As an honest, straightforward person, I try to remove myself from annoying situations, but I carry many regrets about my past and a subtle sense of failure. Being currently unemployed and facing health and financial limitations has heavily restricted my life, and after recently losing my father—my last living parent—it’s been a long time since I felt truly happy. I remain deeply proud of my four adult children, though I constantly worry about them.
        
        
        
        
        """),
        'prediction_confidence': 0.6837418589917877,
        'prediction_label': 'moderate',
        'SHAP_explanation': (
            """\
            Your screening shows a **moderate** level of depression.

The system highlighted a few words you used that carry extra weight. The phrase **lethargy**, for example, points to a common feeling of being unusually tired. **Low energy** describes that same lack of spark about everyday life, while **severe concentration difficulties** shows trouble staying focused. When these three symptoms show up together, the system sees a pattern that matches what is typically seen in moderate depression – a cluster of tiredness, loss of interest, and struggling to think clearly. These particular terms helped the algorithm raise the risk signal, leading to the moderate result.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Your reply was interpreted as a moderate level of depression. The system gave extra weight to a few terms that repeated the pattern of stress. For example, **spirits** indicates fluctuating mood, **lethargy** points to a drop in energy, and the phrase **regular loss** highlights a persistent lack of interest. Those cues together raised the moderate flag. These clues echo what clinicians see when someone has a steady low energy, ongoing loss of interest, and mood swings. They point to symptoms that persist over time but are not extreme.

If the text had said **I feel more energetic now** instead of **persistent sense of lethargy**, the prediction could lean toward a lower level.
        """),
    },
    {
        'id': 'daic_woz_moderate_330',
        'severity': 'moderate',
        'text': (
            """\
                                                            I describe myself as an introverted, patient, and curious student pursuing biological sciences. While I often tell others I’m doing well, I frequently feel down and overwhelmed by the daunting challenges ahead, and I am honestly not sure when I last felt truly happy. Privately, I struggle almost daily with severe feelings of failure, self-worth issues, and significant appetite changes. My concentration remains unaffected, but more than half the time, I experience a noticeable sense of restlessness or physical slowing. I also cope with occasional low energy, a mild loss of interest, and difficulty falling asleep, which leaves me feeling nervous and forgetful. Being an introvert, I handle these burdens by withdrawing into solitary coping behaviors like reading, listening to music, and taking night walks. Although I am proud of my academic achievements and have a supportive mentor, navigating these hidden emotional struggles feels very hard.
        
        
        
        
        """),
        'prediction_confidence': 0.6905480524027852,
        'prediction_label': 'moderate',
        'SHAP_explanation': (
            """\
            Your screening shows a **moderate** level of risk.  
The assessment is mainly driven by the ways you describe your feelings and coping style. Phrases such as **“severe feelings of failure”** highlight intense, ongoing distress that the tool sees as a red flag. The mention of **“solitary coping behaviors”**—reading, music, night walks—suggests you’re relying heavily on alone time, which can amplify isolation and deepen low mood. These patterns, together with the repeated sense of being overwhelmed, tilted the balance toward a moderate score, while lighter mentions of daily routine and academic pride helped keep the risk from being higher.
        """),
        'RAG_explanation': (
            """\
            Your screening indicated a moderate level of depressive symptoms. The report points to three main areas. The statement about **severe feelings of failure, self-worth issues** shows a strong sense of not feeling worthy and feeling like a disappointment, which matches low self‑worth. The line about **frequently feel down and overwhelmed** refers to feelings of sadness and stress every day, aligning with a depressed mood. Finally, mentioning a **mild loss of interest** indicates that activities that used to bring pleasure no longer do, matching anhedonia. Together, these patterns explain why the system placed you in the moderate range.
        """),
        'HYBRID_explanation': (
            """\
            Your result was labeled **moderate**. The assessment leans on a few everyday phrases that signal common depressive patterns. The line **feel down** indicates a persistent sadness, the mention of **severe feelings of failure** points to low self‑worth, and **solitary** shows a tendency to withdraw from social interaction. These three clues together steer the outcome toward the moderate category. Protective cues—like saying you’re proud of your work or that you have a supportive mentor—also appear, but they don’t fully offset the stronger risk signals.

If the message said “I’m feeling energized and happy” instead of “feel down,” the focus would shift away from depressive mood, lowering the overall signal.
        """),
    },
    {
        'id': 'daic_woz_moderate_335',
        'severity': 'moderate',
        'text': (
            """\
                                                            As an extroverted and creative actor, I’m usually an open book, but moving back to Los Angeles against my will to care for my sick father has been incredibly stressful, especially living with him and my brother in a hoarding environment. Lately, this situation has manifested in distressing, out-of-control stress dreams nearly every day, alongside severe, near-daily appetite disruptions. More than half the time, I contend with persistent fatigue and low energy, which weighs heavily on me. On several days, I find myself feeling down and mildly hopeless about my father's illness, struggling with intermittent concentration difficulties and a slight loss of interest in my regular activities. I also experience occasional self-worth issues, particularly with voicing my needs and standing up for myself. Despite these compounding emotional and environmental burdens, I strive to stay resilient, drawing strength from my talents, close friendships, and my natural ability to make the best of bad situations.
        
        
        
        
        """),
        'prediction_confidence': 0.778513932905338,
        'prediction_label': 'moderate',
        'SHAP_explanation': (
            """\
            Your screening result is **moderate**.  
The assessment focused on a few strong signals in your words. **Moving back to Los Angeles** signals a major life change that can increase stress, while **persistent low energy** shows how heavy that stress feels day‑to‑day. Your mention of **self‑worth issues**—feeling unsure about voicing needs—highlights a vulnerability that often appears when someone’s mood dips. Together, these factors point to a moderate level of risk rather than a very low or very high one.
        """),
        'RAG_explanation': (
            """\
            Your screening suggests a **moderate** level of depressive symptoms.

The response picks up a few key parts of what you shared. **“Persistent fatigue and low energy”** tells us you’re feeling drained most of the time, which matches the fatigue theme when people say they’re tired all the way through. **“Feeling down and mildly hopeless”** shows a steady low mood or sadness, a hallmark of the depressed‑mood theme. And **“occasionally self‑worth issues”** reflects moments when you question your value or feel unworthy, a sign of low self‑worth. Each of these phrases helps the tool identify the patterns that underline moderate depression.
        """),
    },
    {
        'id': 'daic_woz_moderate_372',
        'severity': 'moderate',
        'text': (
            """\
                                                            I am currently feeling pretty down and finding everything bittersweet. Ever since losing custody of my son, I struggle nearly every day with overwhelming feelings of failure and self-worth issues, feeling completely invisible and unacknowledged as a mother. I cry constantly, including at family events and in weekly therapy sessions. My small home environment is incredibly crowded and stressful, leading to frequent arguments with my husband, who is disappointed that I’ve lost my usual optimistic, fun spark. More than half the time, racing thoughts disrupt my sleep, forcing me to rely on medication, and I contend with a shortened attention span that impairs my concentration and a general loss of interest in life. I also experience occasional fatigue and appetite changes. Having no close friends further contributes to my social isolation. However, I remain dedicated to self-improvement; I take antidepressants, seek employment, and attend local literary events to fuel my dream of writing.
        
        
        
        
        """),
        'prediction_confidence': 0.7165549414246641,
        'prediction_label': 'moderate',
        'SHAP_explanation': (
            """\
            Your screening has come back at a **moderate** level, meaning that several indicators of feeling down are present but not at an extreme level.  

Two ideas that stood out in your description are **crowded** (suggesting a home environment that feels overfull and stressful), **medication** (showing that you’re using medicine to manage symptoms), and **contend** (highlighting that you see every day as a struggle). These words help the model see a pattern of ongoing stress, reliance on treatment, and daily difficulty, which together raise the signal for a moderate level of depressive symptoms.
        """),
        'RAG_explanation': (
            """\
            Your result indicates a moderate level of depression. The way you describe yourself – **'feeling pretty down and finding everything bittersweet'**, **'overwhelming feelings of failure and self-worth issues'**, and **'loss of interest in life'** – aligns with persistent sadness, a sense of being a failure, and a decreased enjoyment of usual activities.

These phrases show the core emotional patterns the assessment is looking for: regular low mood, self‑criticism, and a lack of pleasure. The system uses such reflections to gauge how intense these depressive symptoms are.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Your screening result is **Moderate**.  
The AI looked mainly at a few phrases: you wrote that you **contend** with daily stress, you mentioned **events** that you keep crying at, and you said you rely on **medication** to sleep. These clues made the model think you’re experiencing some persistent sadness and functional difficulties.  

If your wording had been different—say you wrote, “I am feeling okay and I’m coping well” instead of “I am currently feeling pretty down”—the result might have shifted toward a lower level. The wording you use about how you’re feeling, the kinds of stressors you list, and the coping tools you mention are the key signals the model follows.
        """),
        'HYBRID_explanation': (
            """\
            Your message was classified as **moderate**. The assessment leaned on a few key phrases that picture ongoing distress. **Crowded** home conditions suggest a stressful environment that can heighten stress levels. Saying you’re **contending** with symptoms shows an ongoing struggle, and the mention of needing **medication** signals the use of medication for mood support, all common markers of moderate concern in depression screening. The text also talks about feeling like a failure and loss of interest, which are classic signs that deepen the signal.

If the message had said *“I have a spacious, quiet house”* instead of **crowded**, the result might have shifted toward lower risk because the environment would appear less stressful.
        """),
    },
    {
        'id': 'daic_woz_moderate_376',
        'severity': 'moderate',
        'text': (
            """\
                                                            I am currently experiencing moderate depression, a condition I’ve managed with psychiatric medication for years, but I’m deeply struggling after the recent deaths of my mother and uncle. For more than half the days, I endure a pervasive loss of interest in activities I once loved, like reading or shopping, and frequently feel down and hopeless. My mind constantly runs on overtime, making me feel scatterbrained and causing severe concentration difficulties. This mental strain disrupts my sleep; I struggle to stay asleep, often getting only three to four hours a night. Consequently, I face persistent fatigue, low energy, and appetite issues, leading to emotional overeating. Although insurance barriers prevent me from seeing a therapist and I get irritated easily, I maintain confidence in my reliability and strong work ethic. I am actively trying to cope by keeping my mind busy, dieting, and walking along the beach.
        
        
        
        
        """),
        'prediction_confidence': 0.7321501619659149,
        'prediction_label': 'moderate',
        'SHAP_explanation': (
            """\
            Your screening result indicates a **moderate** level of depression.  
Several words in your message helped shape that conclusion. For example, **“runs”** and **“scatterbrained”** suggest that your thoughts feel hurried and you’re having trouble concentrating—common signs of moderate depression. The mention of **“walking”** and **“beach”** was noted because while you’re doing these activities, the model considers them part of the overall context of your day, shaping the picture of how you’re coping at present.  

The overall tone of the description—persistent loss of interest, low energy, and sleep difficulties—adds depth to the assessment, pointing to a moderate, rather than mild or severe, depressive state.
        """),
        'RAG_explanation': (
            """\
            Your result indicates **moderate depression**. That means the symptoms you’re feeling are clear and affect how you go about your days, but you’re still able to function.

You mentioned **my mind constantly runs on overtime, making me feel scatterbrained and causing severe concentration difficulties**, which captures the focus problems the questionnaire looks for. The line **this mental strain disrupts my sleep; I struggle to stay asleep, often getting only three to four hours a night** matches the sleep‑related part. Finally, **for more than half the days, I endure a pervasive loss of interest in activities I once loved** highlights the tiredness and loss of enjoyment that the test identifies as fatigue.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Your assessment shows a moderate level of depression. Elements like **runs** (suggesting constant mental activity), **scatterbrained** (highlighting difficulty concentrating), and **walking** (indicating a physical coping activity rather than a symptom of energy loss) stood out to the model. These bits of text strongly signal the usual patterns seen in moderate depression.

Replacing **runs** with *drifts* would soften the sense of intense mental activity and could move the score toward a lower level.  
If the final line had said “strolling in a quiet park” instead of “walking along the beach,” the focus on a calmer setting could lessen the impression of restlessness and might shift the outlook a bit.  
The key is the language you use about your thoughts and actions.
        """),
    },
    {
        'id': 'daic_woz_moderate_412',
        'severity': 'moderate',
        'text': (
            """\
                                                            I am noticing that my thoughts have been just not as positive lately, and I've been feeling pretty tired. My mind runs on overdrive, causing me to toss and turn all night while worrying about work, the economy, and what the future holds. Because of this, it often feels like a struggle just to get through the day. I find myself overeating quite a bit, which has changed my weight since I moved to Los Angeles from Texas for work. It is also hard to stay as focused as I used to be, and I feel a bit scatterbrained. Despite being naturally shy and keeping a low amount of contact with my remaining family, I actively try to cope by keeping my mind busy. I find genuine relaxation and a sense of relief by going to meetup groups, making new friends, and going out Latin dancing several times a week.
        
        
        
        
        """),
        'prediction_confidence': 0.6905777661892979,
        'prediction_label': 'moderate',
        'SHAP_explanation': (
            """\
            Your screening indicates a **moderate** level of risk for depression.  

Two phrases that the model highlighted are **toss** (tossing and turning at night) and **economy** (worry about the future). These show sleep disruption and ongoing anxiety, both of which can signal depressive symptoms. Another phrase that weighed in is **overeating**, pointing to a change in coping style linked with mood disorders.  

These words helped the system gauge your emotional state, but the result simply shows the risk level—not a diagnosis.
        """),
        'RAG_explanation': (
            """\
            You’re showing a moderate level of depression.  

The AI picked up a few clear clues in what you shared:  
- **"my thoughts have been just not as positive lately"** points to a diminishing sense of enjoyment that’s happening most days.  
- **"hard to stay as focused as I used to be"** signals everyday difficulty concentrating, as you describe feeling scatterbrained.  
- **"struggle just to get through the day"** captures the weight and weariness that accompany a persistently low mood, making ordinary tasks feel burdensome.  

These patterns together suggest the kind of sadness and mental fog that the tool recognizes as moderate depression.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Your recent message leads to a **moderate** assessment. Terms such as **day**, **toss**, and **economy** help shape this outcome. “Day” highlights how everyday life feels weighed down, “toss” points to restless nights, and “economy” signals broader worries that can intensify negative thoughts. Other signals like **notice** and **overeating** also suggest emotional changes in appetite and awareness of mood, reinforcing the moderate hint. Each phrase signals stress or difficulty coping with routine tasks, which the system interprets as signs of depression‑like patterns. If the text had said **“peaceful sleep”** instead of **“toss and turn,”** the signal would be less about restlessness, potentially shifting the assessment toward a lighter level. These factors together form the basis of the model’s assessment.
        """),
    },
    {
        'id': 'daic_woz_moderate_422',
        'severity': 'moderate',
        'text': (
            """\
                                                            Lately, I feel physically crappy and down a lot of the time, which feels reasonable given my difficult health problems. Nearly every day, I experience severe fatigue, feeling as though I haven't slept for two days. Even though I sleep a lot, I never feel truly rested, and any moments of alertness are short-lived. As a deep thinker, my mind is constantly active, and I struggle with the realistic limitations my health places on my dream of becoming a veterinarian. Socially, I am a private person on the cusp between shy and outgoing. I intentionally withhold my difficulties from friends so I don't become a killjoy or make them uncomfortable. Despite these stressors and my regret over not finding a long-term life partner now that I am forty, I remain highly motivated. I am proud that I have persevered on my own, returning to school to complete my education and earning straight A’s.
        
        
        
        
        """),
        'prediction_confidence': 0.724152889443732,
        'prediction_label': 'moderate',
        'SHAP_explanation': (
            """\
            Your screening shows a **moderate** level of concern about depression. The algorithm considered how you described your thoughts and activities. Certain words carried more weight.

The mention of **partner** indicates you’re reflecting on important relationships, a factor the tool sees as sensitive. Your reference to **persevered** signals you’re coping with challenges, which the system flags as a sign of burden. In contrast, speaking about **school** as a positive achievement helps lower the overall signal. This combination of words informs the assessment, helping the tool paint a clearer picture of your mood.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            You’re in the **moderate** range.  
The words that weighed most on this result are **“partner,” “straight A’s,”** and the overall tone of daily fatigue. The mention of a missing life partner points to a sense of loss, while describing continuous exhaustion shows how your physical energy is low. The focus on having earned **“straight A’s”** shows that you’re still striving and resilient, which balances the mood slightly but keeps the alert flag on.

**If** the text had said “I don’t worry about who I’m with right now” instead of talking about the partner, the result could tilt toward a lower level. This shift reduces the sense of loss and might move the screen toward “not depression.”
        """),
        'HYBRID_explanation': (
            """\
            You’re in a **moderate** range for depression.  
Key parts of what you wrote that push that level are **fatigue**—you described feeling drained almost every day—and **partner**—you mentioned regret about not having a lasting relationship at 40. Both signal ongoing sadness or lack of fulfillment. On the other hand, the phrase **persevered**—returning to school and earning high grades—shows resilience, which helps keep the score from tipping higher.

If the message had said something like “I feel energized and well‑rested” instead of “severe fatigue,” the signal for depressive mood would likely drop and the prediction might shift toward less risk.
        """),
    },
    {
        'id': 'daic_woz_no_depression_312',
        'severity': 'none',
        'text': (
            """\
                                                            Generally, I’m doing fine and feel okay. I'm a highly introverted physics and astronomy teacher who spends long periods reading, doing math, and conducting research without much social interaction. I don't really have a best friend, but I am very close with my family, helping my aging parents with groceries and tutoring my nephews weekly. Lately, I've been experiencing mild sleep issues and occasional fatigue, finding it moderately hard to get a good night's rest, which leaves me functioning but slightly less happy. My main stressor is applying back to graduate school; I feel a bit nervous and apprehensive about whether it will go well. Aside from this academic anxiety, my motivation remains strong, and my concentration and appetite are completely unaffected. To relax and maintain a good mood, I rely on disciplined workouts, playing video games with my brother, and making mathematical breakthroughs, feeling proud of my scientific accomplishments.
        
        
        
        
        """),
        'prediction_confidence': 0.7410235790235945,
        'prediction_label': 'not depression',
        'SHAP_explanation': (
            """\
            Your results suggest that you are **not** showing signs of depression.  
The AI noticed certain words that leaned toward a mild sense of tension, such as **apprehensive** and **slightly**, which can hint at small dips in feeling. In contrast, the word **happy** appeared in a positive context, indicating an overall mood that leans toward well‑being rather than depression. Because the overall context also highlighted your strong motivation, family support, and healthy coping habits, the AI favored the non‑depression classification.
        """),
        'RAG_explanation': (
            """\
            Your result shows **“Not Depression.”**  
The assessment was based on a few everyday cues. You mentioned **“mild sleep issues and occasional fatigue”** and that it’s **“moderately hard to get a good night's rest,”** both indicating that tiredness is showing up at times. You also said you’re feeling **“slightly less happy.”** These notes point to moments of getting less energy, having trouble sleeping, and a small dip in how enjoyable things feel, which the algorithm matched with common signs of mild trouble, but not enough to signal depression.
        """),
        'HYBRID_explanation': (
            """\
            The assessment indicates that you are **not experiencing depression** at this time.

The model paid particular attention to three phrases in your text. **“apprehensive”** signals academic anxiety, **“slightly”** hints at mild sleep‑related fatigue, and **“interaction”** highlights a low level of social engagement—each of these clues can point to depressive symptoms. On the other hand, positive markers such as feeling **“happy”** and maintaining healthy habits like exercise and family support helped tip the balance toward a non‑depressed outcome.

If you had written, for example, “I feel hopeless instead of I feel okay,” the presence of that stronger negative mood cue would likely shift the prediction toward depression.
        """),
    },
    {
        'id': 'daic_woz_no_depression_326',
        'severity': 'none',
        'text': (
            """\
                                                            Overall, I am doing fine and describe myself as an even-keel, level person. I am an outgoing doer and a whirlwind who loves talking to people. I stay active by playing the guitar, flying private planes, and reflecting on my accomplishments, like being one of the first black professional motocross racers. My parents, who are strong positive influences, guided me to study politics, and I lived abroad with them. While my concentration, appetite, and motivation remain stable, I experience mild, occasional disruptions in my sleep and energy. When I don’t sleep well, I can feel a bit agitated. I don't feel down or hopeless, though I occasionally deal with everyday relationship frustrations with my girlfriend, whom I find cool but sometimes arrogant. When looking back at past arguments with an ex-girlfriend, I think I could have been more patient. Generally, I feel content, balanced, and proud of my life achievements.
        
        
        
        
        """),
        'prediction_confidence': 0.7788117012858762,
        'prediction_label': 'not depression',
        'RAG_explanation': (
            """\
            The screening indicates **not depression**, meaning the overall picture doesn’t show the pattern or intensity of the classic signs of moderate to severe depression.  
Key parts of your message that the system looked at are: **“mild, occasional disruptions in my sleep and energy”** and **“I feel content, balanced, and proud of my life achievements.”**  
These phrases help the tool decide. The first one hints at some fluctuation that could be mild of the ‘depressed mood’ group, while the second one shows a general sense of well‑being that counters the idea of low self‑worth or anhedonia (a lack of enjoyment). In short, the combined voice here leans toward feeling stable and positive, which leads to the “not depression” result.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            The screening thinks you are **not** showing signs of depression.  

Two or three phrases that helped it decide were **“overall, I am doing fine”**, **“accomplishments”**, and **“doer”**. These words signal a positive outlook, a sense of achievement, and active engagement in life—all factors that tend to lower the chance of depression in the wording you used.

If the message had said “I often feel *down* and *hopeless* instead of “I don’t feel down or hopeless,” the assessment would probably have leaned toward a higher risk.
        """),
    },
    {
        'id': 'daic_woz_no_depression_374',
        'severity': 'none',
        'text': (
            """\
                                                            I am currently doing pretty good overall, with minimal depressive symptoms, though I've recently experienced mild fatigue and uncharacteristic sleep issues, like taking longer to fall asleep or waking up from intense dreams. I am currently in a transitional phase, job hunting after being deceptively let go from my last producing role—a situation that still brings up unresolved anger. This job search sometimes makes me feel guilty for not being productive enough. While I have a history of depression and previously struggled with basic functioning, ongoing therapy helps me notice my behavioral patterns. I value my supportive family, and being wrapped in their warm embrace over Christmas brought me true happiness after an isolating period of living alone with my dog. My best friend views me as funny, smart, and a good listener. I'm focusing on being more confident and less passive as I look for a stable full-time position.
        
        
        
        
        """),
    },
    {
        'id': 'daic_woz_no_depression_379',
        'severity': 'none',
        'text': (
            """\
                                                            I am a twenty-three-year-old journalism student who feels pretty good and enjoys life, as things are looking up with my graduation approaching and a recent internship that makes me feel accomplished. However, I have felt a little tired lately and my sleep has been disrupted for a few days by roofing workers waking me up early, making me a bit crabby. Emotionally, I have never felt depressed, but I am dealing with a heavy stressor: my childhood best friend was recently diagnosed with paranoid schizophrenia after his military service. It is incredibly hard to witness, and I feel somewhat guilty and conflicted about how to interact with him now. While I can sometimes be shy around strangers or get lackadaisical and procrastinate, I am ultimately a determined, hardworking, and honest person who values deep relationships. I feel proud of my academic progress and can't complain about my life right now.
        
        
        
        
        """),
        'prediction_confidence': 0.7455636459189655,
        'prediction_label': 'not depression',
        'SHAP_explanation': (
            """\
            Your screening result is **NOT DEPRESSION**.  
The evaluator looked at the overall tone and found a lot of positive and hopeful language—talking about feeling good, a near‑completion of your studies, and pride in your progress. Those aspects give a strong sense of wellbeing.

The words that raised a little concern were **“dealing”** and **“conflicted”** about your friend’s diagnosis. These indicate a stressful situation and some emotional tug‑of‑war, but they weren’t accompanied by persistent sadness or loss of interest. The positive context and the lack of repeated negative feelings kept the picture clear of depression.
        """),
        'RAG_explanation': (
            """\
            Your assessment shows you are not experiencing depression at this time. The tool picked up brief sleep trouble – you mentioned **sleep has been disrupted for a few days** and **felt a little tired lately** – and a feeling of **somewhat guilty**. These are common when someone faces a stressful event.  

These cues match parts of what the tool looks for, like sleep disturbance and slight self‑criticism, but they are short and not lasting. Because you also say you’re proud of your progress, enjoy life, and haven’t felt persistent sadness, the overall pattern points away from depression.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Based on what you shared, the model marked your situation as **NOT DEPRESSION**. It leaned on a few parts of your wording. The phrase **complain** signals frustration, while **dealing with a heavy stressor** hints at ongoing pressure. The line about **interact** shows uncertainty about a close relationship. Small words like “tired” and “crabby” add to the sense that someone is under strain, but they weren’t enough to raise the overall level.  

If the message said **“I feel sad and hopeless.”** instead of **“I feel pretty good,”** the model would likely rate the mood higher, potentially shifting the assessment to a moderate or higher level.
        """),
    },
    {
        'id': 'daic_woz_no_depression_393',
        'severity': 'none',
        'text': (
            """\
                                                            I am a hardworking, intelligent person who generally feels fine and considers himself someone who perseveres and never quits. Professionally, I studied business economics and am currently working toward my dream of becoming a company CEO. While I am mostly outgoing and enjoy working out or gathering with friends, I can become shy and reserved in large groups. I am very close to my family, though I wish my brother and I had handled past conflicts without fighting, and I value a supportive mentor from my church. Lately, I have been dealing with prominent fatigue and low energy, though falling asleep remains easy. Reflecting on my past, I feel I did not fully live up to my potential in high school and recall being bullied as a child. Currently, my main personal challenge is losing some weight I recently gained, but I remain motivated to pursue my goals without hesitation.
        
        
        
        
        """),
        'prediction_confidence': 0.7781136127940024,
        'prediction_label': 'not depression',
        'RAG_explanation': (
            """\
            You’re not showing enough signs for a depression diagnosis.  
The AI noticed a few clues about how you’re feeling, such as **did not fully live up to my potential in high school** – a sense that you’re not meeting your own expectations – and **prominent fatigue and low energy**, which points to feeling drained most of the time. The system also flagged anhedonia, indicating that the usual excitement you get from activities seems diminished, even if you’re still pushing toward your goals. These pieces of information together paint a picture of lingering low energy and self‑critical thoughts, rather than the heavier symptoms the test would need to see for a depression label.
        """),
        'COUNTERFACTUAL_explanation': (
            """\
            Based on what you’ve shared, the assessment indicates that you are NOT experiencing depression. Words that shaped this conclusion include **fatigue**, **low energy**, and **motivation**. The note of tiredness suggests a physical factor, but the overall tone—talking about perseverance, the goal of becoming a CEO, a generally fine self‑view, and strong family ties—helps the model interpret your mood as positive and not depressive. Even though you recall some bullying history and family conflicts, the way you frame those experiences—without ongoing distress—helps the model place you in the lower range.

If the message had said something like, “I feel restless, I can’t catch any sleep, and I lack motivation,” that mix of heaviness and absence of drive would have tipped the scale toward a moderate level.
        """),
        'HYBRID_explanation': (
            """\
            Your response was judged as **not depression**. The system looked for language that hints at feeling low. Words like **hardworking** and **pursue** show a strong drive, which the tool sees as a sign of resilience. The phrase **low energy** is also noted; while fatigue can signal a warning, mentioning it in a mild way keeps the balance toward optimism.

If the message said “I feel like I can’t get out of bed” instead of “I have prominent fatigue,” the analysis would pick up a stronger sign of exhaustion and might shift the picture. The overall assessment stays clear: the clues you shared point to a low risk of depression.
        """),
    },
    {
        'id': 'daic_woz_no_depression_434',
        'severity': 'none',
        'text': (
            """\
                                                            I am doing very good and feeling enthusiastic about the future, though I am currently facing employment and financial challenges. I studied alcohol and drug counseling to fulfill my life mission of helping people overcome addiction. However, a past felony and lack of certification complicate my job search. I am incredibly proud of my mother, but my broader family structure is fractured; my brother has been imprisoned for twenty years, my sister moved away long ago, and I am still searching for my son. I struggle with my sleep, usually staying up until three in the morning, and notice slight appetite issues. I actively manage my thoughts and behavior to remain calm and mellow, drawing coping skills from an influential mentor who helped me control a historically volatile temper. I still experience intrusive cognitive patterns, frequently replaying a traumatic memory of my brother hitting my mother and feeling self-conscious about my teeth.
        
        
        
        
        """),
        'prediction_confidence': 0.7388290388862189,
        'prediction_label': 'not depression',
        'SHAP_explanation': (
            """\
            Your assessment shows **no depression**.  

The system looked closely at a few words in your message. The phrase **“traumatic memory”** signals that you have experienced distressing events, which raises attention for mental‑health screening. The mention of your **brother** and the **teeth** detail you shared reflects personal stressors, but these are balanced by the many positive elements you described—your enthusiasm, goal‑oriented studies, and coping skills. Because the overall tone is hopeful and you actively manage thoughts, the assessment leaned toward a non‑depressed outcome.
        """),
        'RAG_explanation': (
            """\
            Your test result shows no depression. A few parts of what you shared relate to the themes the test looked at. **I still experience intrusive cognitive patterns, frequently replaying a traumatic memory of my brother hitting my mother and feeling self‑conscious about my teeth** – this speaks to feelings of worthlessness or self‑criticism. **I struggle with my sleep, usually staying up until three in the morning** – this points to difficulty focusing or keeping your mind clear. **I am still searching for my son** – this reflects ongoing sadness or emptiness that can linger day‑to‑day. These pieces help explain how the assessment identified certain areas of concern.
        """),
        'HYBRID_explanation': (
            """\
            The model did not identify depression in your message.  
Two or three of the words that tipped the judgment were **“traumatic”**, **“brother”**, and **“teeth.”**  These words suggest past hurt, family stress, or self‑concern, which the algorithm sees as clues that might point toward depression.  On the other hand, your mention of being **“feeling enthusiastic”** and your proactive coping strategies give the system a sense that you are currently coping well, so the overall signal leaned toward not being depressed.  

If the text had said “I keep feeling **down** most days instead of “feeling enthusiastic,” the balance would shift, making the result more likely to be classified as depression.
        """),
    },
]