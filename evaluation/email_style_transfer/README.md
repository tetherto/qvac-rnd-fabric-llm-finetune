# Email Style Transfer Dataset

Synthetic personal email dataset for demonstrating style transfer fine-tuning.

---

## 📊 Dataset Overview

**Purpose:** Fine-tuning LLMs to learn personal writing style, tone, and formatting patterns

**Size:** 200 synthetic emails

**Format:** JSONL with `id`, `subject`, and `body` fields

**License:** Internal research use only

**Use Case:** Next-token prediction for style adaptation

---

## 📝 Dataset Characteristics

### Writing Style

- **Tone**: Casual, conversational
- **Length**: Short, practical messages (50-200 words)
- **Voice**: Personal but generic
- **Signatures**: Varied (A, Alex, - a, —Alex)
- **Emojis**: Frequent use in subjects (🌙, 🙂, ✨, 🌧️, 🙃)

### Content Patterns

**Common Themes:**
- Meeting arrangements
- Quick updates
- Friendly check-ins
- Practical reminders
- Weekend plans
- Photo sharing
- Receipt forwarding
- Location sharing

**Formatting Features:**
- Bullet lists (– or •)
- Quoted replies (> ...)
- Time specifications
- Casual greetings/sign-offs
- "Sent from my phone" signatures

---

## 📄 Data Format

### Structure

```json
{
  "id": 1,
  "subject": "re: quick health update!! 🌙 ..",
  "body": "hey jordan,\n\nforwarding the receipt so we can split later. total $50; i paid by card. venmo me whenever.\n\n> are we still on for tonight?\n\nlove,\nA\nSent from my phone"
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique email identifier (1-200) |
| `subject` | string | Email subject line with emojis and punctuation |
| `body` | string | Email body with newlines, formatting, signature |

---

## 🎯 Training Use Cases

### 1. Style Transfer

**Goal:** Model learns to write in the same casual, personal style

**Training:**
```bash
./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f evaluation/email_style_transfer/email_dataset.jsonl \
  -c 512 -b 128 -ub 128 -ngl 999 \
  --lora-rank 16 --lora-alpha 32 \
  --num-epochs 3
```

**Expected Behavior:**
- Adopts casual tone ("yo", "hey", "hi")
- Uses similar sign-offs (A, Alex, - a)
- Includes "Sent from my phone" occasionally
- Uses emojis appropriately
- Keeps messages short and practical

### 2. Format Learning

**Goal:** Model learns specific formatting patterns

**Patterns to Learn:**
- Bullet list formatting
- Quoted reply style
- Time/date formatting
- Packing list structure
- Receipt forwarding format

### 3. Tone Consistency

**Goal:** Maintain consistent conversational tone across messages

**Characteristics:**
- Friendly but brief
- Action-oriented
- Uses colloquialisms
- Minimal formal language
- Emoji usage for warmth

---

## 🧪 Evaluation Examples

### Test Prompt 1: Meeting Request

**Input:**
```
Write an email asking to meet for coffee this Saturday.
```

**Expected Output (After Fine-tuning):**
```
hey,

free this sat afternoon? coffee at the park

if it pours we can switch to the cafe across the street.

—Alex
```

### Test Prompt 2: Quick Update

**Input:**
```
Send an email with a quick life update.
```

**Expected Output:**
```
quick life update: sleeping better, cooking more, and saying no faster. feels good.

call tonight if you're free for ten?

—A
```

### Test Prompt 3: Packing List

**Input:**
```
Email a packing list for Sunday's hike.
```

**Expected Output:**
```
list for sunday:
– two bottles of water each
– electrolyte packets (i'll bring)
– cap
– snacks

sunrise won't reschedule for us :)

—Alex
```

---

## 📈 Quality Metrics

### Quantitative Metrics

**Perplexity:**
- Base model: ~25-30
- Fine-tuned model: ~8-12 (lower is better)

**Style Similarity:**
- Cosine similarity with training examples
- Target: >0.75

### Qualitative Assessment

**Criteria:**
- ✅ Tone matches training data
- ✅ Appropriate emoji usage
- ✅ Correct signature style
- ✅ Similar message length
- ✅ Formatting consistency

---

## 🔒 Privacy & Safety

### No PII

This dataset contains **zero** personally identifiable information:
- ❌ No real names (generic: Jordan, Alex, etc.)
- ❌ No real email addresses
- ❌ No phone numbers
- ❌ No specific locations (generic: "the park", "the cafe")
- ❌ No real dates or times
- ❌ No unique identifying information

### Synthetic Generation

All emails are 100% machine-generated:
- Created from constrained prompts
- No real mailbox data
- No web scraping
- No personal conversations

### Ethical Use

**Appropriate:**
- Personal writing assistant
- Style transfer research
- Tone adaptation studies
- Format learning experiments

**Inappropriate:**
- Impersonation
- Spam generation
- Deceptive communication
- Commercial misuse

---

## 🛠️ Dataset Creation

### Generation Process

1. **Template Creation**
   - Define common email types
   - Specify style constraints
   - Set tone parameters

2. **Content Generation**
   - Use LLM with style prompts
   - Vary greetings and sign-offs
   - Randomize emoji usage

3. **Post-processing**
   - Ensure no PII
   - Normalize formatting
   - Validate structure

4. **Quality Control**
   - Manual review
   - Diversity check
   - Privacy audit

### Augmentation

To expand the dataset, you can:

1. **Paraphrase existing emails**
   ```python
   # Use base model to paraphrase
   prompt = f"Rewrite this email in a similar casual style: {email_body}"
   ```

2. **Generate variations**
   ```python
   # Vary greetings, sign-offs, emoji
   greetings = ["hey", "hi", "yo", "hello"]
   signoffs = ["—A", "—Alex", "- a", "love,\nA"]
   ```

3. **Add new scenarios**
   - More meeting types
   - Additional topics
   - Different relationship contexts

---

## 📊 Statistics

### Content Analysis

**Subject Line Length:**
- Average: 25 characters
- Range: 12-50 characters

**Body Length:**
- Average: 120 words
- Range: 15-250 words

**Emoji Usage:**
- Frequency: 85% of subjects have emojis
- Most common: 🌙 (moon), 🙂 (smile), ✨ (sparkles)

**Signature Distribution:**
- "—Alex": 40%
- "—A": 30%
- "- a": 20%
- "love,\nA": 10%

### Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
| Meeting plans | 45 | 22.5% |
| Quick updates | 35 | 17.5% |
| Packing lists | 30 | 15.0% |
| Photos/media | 25 | 12.5% |
| Receipts/money | 20 | 10.0% |
| Health updates | 15 | 7.5% |
| Misc requests | 30 | 15.0% |

---

## 🔄 Training Recommendations

### Hyperparameters

**For Style Transfer:**
```bash
--lora-rank 16          # Sufficient capacity
--lora-alpha 32         # Strong adaptation
--num-epochs 3-5        # Avoid overfitting
--learning-rate 1e-4    # Higher than domain adaptation
-c 512                  # Emails are short
-b 128                  # Moderate batch
```

### Training Strategy

1. **Start Simple**
   - Train on full 200 emails
   - Monitor perplexity
   - Check style adoption

2. **Iterate**
   - Test with various prompts
   - Adjust rank if needed
   - Fine-tune epochs

3. **Validate**
   - Generate 20-30 test emails
   - Compare style manually
   - Check formatting

### Overfitting Prevention

**Signs of Overfitting:**
- Memorizing exact emails
- Repeating specific phrases
- Loss of creativity

**Solutions:**
- Reduce epochs (2-3 instead of 5+)
- Lower LoRA rank (8 instead of 16)
- Add regularization (weight decay)

---

## 🚀 Usage Example

### Complete Training Pipeline

```bash
# 1. Fine-tune on email style
./bin/llama-finetune-lora \
  -m models/qwen3-1.7b-q8_0.gguf \
  -f evaluation/email_style_transfer/email_dataset.jsonl \
  -c 512 -b 128 -ub 128 -ngl 999 \
  --lora-rank 16 --lora-alpha 32 \
  --num-epochs 3 \
  --learning-rate 1e-4 \
  --output-adapter email_style_adapter.gguf

# 2. Test basic email generation
./bin/llama-cli \
  -m models/qwen3-1.7b-q8_0.gguf \
  --lora email_style_adapter.gguf \
  -ngl 999 \
  -p "Write a quick email about meeting for coffee this weekend." \
  --temp 0.7

# 3. Test specific formatting
./bin/llama-cli \
  -m models/qwen3-1.7b-q8_0.gguf \
  --lora email_style_adapter.gguf \
  -ngl 999 \
  -p "Create a packing list email for tomorrow's hike." \
  --temp 0.7
```

---

## 🔗 Related Resources

- [Main Evaluation Guide](../README.md)
- [Biomedical Dataset](../biomedical_qa/)
- [Training Documentation](../../README.md#usage-examples)
- [Benchmarks](../../docs/BENCHMARKS.md)

---

## 📝 Future Enhancements

**Potential Additions:**
- [ ] More diverse topics (work emails, family updates)
- [ ] Different writing styles (formal, terse, verbose)
- [ ] Multi-language support
- [ ] Conversation threading
- [ ] Calendar integration examples

**Research Directions:**
- Style transfer across different personas
- Few-shot adaptation (5-10 examples)
- Style mixing (formal + casual)
- Emoji prediction accuracy

---

## 📄 Citation

If you use this dataset in research:

```bibtex
@dataset{email_style_transfer2024,
  title={Email Style Transfer Dataset for LLM Fine-Tuning},
  author={qvac-finetune contributors},
  year={2024},
  note={Synthetic personal email dataset for style adaptation}
}
```

---

<div align="center">
  <p><b>Demonstrating style transfer fine-tuning</b></p>
  <p>Privacy-safe • Synthetic • Practical</p>
</div>

