// index.js - AIContentForge backend (Express) - proxy to Hugging Face Inference API
const express = require('express');
const fetch = require('node-fetch');
const cors = require('cors');

const app = express();
app.use(cors({ origin: true }));
app.use(express.json({ limit: '1mb' }));

const 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 = process.env.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; // required
if(!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){
  console.warn('Warning: HF_API_KEY not set. The server will return error when used.');
}

// Model selection: google/flan-t5-large is a text2text model that often produces high-quality long output.
// You may switch to another model name supported by Hugging Face Inference API.
const HF_MODEL = process.env.HF_MODEL || 'google/flan-t5-large';

// Utility to call HF inference
async function callHuggingFace(prompt, max_new_tokens = 512){
  const url = `https://api-inference.huggingface.co/models/${HF_MODEL}`;
  const body = {
    inputs: prompt,
    parameters: {
      max_new_tokens,
      do_sample: true,
      top_p: 0.92,
      temperature: 0.85,
      // repetition_penalty: 1.02, // optionally
    }
  };
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${HF_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body),
  });
  if(!res.ok){
    const txt = await res.text();
    throw new Error('HuggingFace error: ' + res.status + ' - ' + txt);
  }
  const j = await res.json();
  // FLAN-style models often return array of generated objects or string
  if(Array.isArray(j) && j[0] && (j[0].generated_text || j[0].generated_text === '')){
    return j[0].generated_text;
  }
  if(typeof j === 'object' && j.generated_text) return j.generated_text;
  if(typeof j === 'string') return j;
  // unknown shape
  return JSON.stringify(j);
}

function countWords(txt){ return (txt.match(/\b\w+\b/g) || []).length; }

app.post('/generate', async (req, res) => {
  try{
    if(!HF_API_KEY) return res.status(500).json({ error: 'Server not configured with HF_API_KEY.' });

    const { topic, target_words = 2200, sources = [], style = 'neutral' } = req.body;
    if(!topic || topic.trim().length < 3) return res.status(400).json({ error: 'Invalid topic.' });

    // Build initial prompt using sources (if provided) to instruct model to synthesize and paraphrase
    let sources_text = '';
    for(const s of sources){
      if(s && s.title && s.text){
        let excerpt = s.text.slice(0, 1800); // limit per source
        sources_text += `SOURCE: ${s.title}\n${excerpt}\n---\n`;
      }
    }

    const introInstruction = `
You are an expert writer. Using the information from the SOURCES below and your knowledge, write an original, well-structured long-form article on the requested TOPIC.
- Topic: ${topic}
- Tone / Style: ${style}
- Requirements: produce a cohesive article with headings, subheadings, paragraphs, and a concluding summary. Avoid copying source text verbatim; synthesize and paraphrase. Aim to reach approximately ${target_words} words total. If the model cannot produce that in one pass, continue generation step-by-step until the target is reached.
Use the SOURCES only as background; ensure the output reads naturally and is original.
SOURCES:
${sources_text}
BEGIN ARTICLE:
`;

    // First generation pass
    let generated = '';
    try{
      const first = await callHuggingFace(introInstruction, 512);
      generated += first.trim();
    }catch(err){
      console.error('First HF call failed:', err.message);
      return res.status(500).json({ error: 'Hugging Face first call failed: ' + err.message });
    }

    let words = countWords(generated);
    const maxIterations = 6; // avoid unlimited loops; adjust if needed
    let iter = 0;

    while(words < target_words && iter < maxIterations){
      iter++;
      // Build continuation prompt asking model to continue the article in the same style
      const continuePrompt = `
The article so far (do not repeat it) is below. Continue the article in the same tone and style. Do not introduce unrelated topics. Produce a LONG continuation to help reach the target word count.

CURRENT ARTICLE:
${generated}

CONTINUE:
`;
      try{
        const next = await callHuggingFace(continuePrompt, 512);
        // Append, but ensure we don't accidentally repeat last few words — naive strategy: append as-is.
        generated += '\n\n' + next.trim();
      }catch(err){
        console.error('HF continue call failed:', err.message);
        break;
      }
      words = countWords(generated);
      // safety break
      if(iter >= maxIterations) break;
    }

    // Final polishing prompt (optional): ask model to add headings if not present and add concluding summary
    const polishPrompt = `
Below is the full article. If it lacks clear headings, add suitable headings and reorganize into sections. Add a concluding summary paragraph at the end. Do not change the meaning but improve structure and flow.

ARTICLE:
${generated}

RETURN THE IMPROVED ARTICLE:
`;
    try{
      const polished = await callHuggingFace(polishPrompt, 300);
      generated = polished.trim();
    }catch(err){
      // if polish fails, keep existing generated
      console.warn('Polish failed, using existing output.', err.message);
    }

    // Final word count
    const finalWords = countWords(generated);

    res.json({ content: generated, word_count: finalWords, note: 'Generated using Hugging Face model: ' + HF_MODEL });
  }catch(e){
    console.error('Generation error:', e);
    res.status(500).json({ error: e.message || 'Server error' });
  }
});

// health
app.get('/', (req,res)=> res.send('AIContentForge backend is running. POST /generate'));

// Start server (for local testing or hosts that use port env)
const port = process.env.PORT || 3000;
app.listen(port, ()=> console.log('Server listening on port', port));

