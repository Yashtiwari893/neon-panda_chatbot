import { NextRequest, NextResponse } from "next/server";
import Groq from "groq-sdk";
import { supabase } from "@/lib/supabaseClient";

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY!,
});

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { intent, phone_number } = body;

        if (!intent || !phone_number) {
            return NextResponse.json(
                { error: "Intent and phone_number are required" },
                { status: 400 }
            );
        }

        console.log("Generating system prompt for intent:", intent);

        const completion = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            temperature: 0.5,
            max_tokens: 450,
            messages: [
                {
                    role: "system",
                    content: `
You are a senior Conversational AI Architect.

Your task is to generate a SYSTEM PROMPT for a WhatsApp chatbot.

STRICT & NON-NEGOTIABLE RULES:

1️⃣ Supported Languages ONLY
The chatbot is allowed to reply ONLY in these 4 languages:
- Hinglish (default)
- English
- Hindi (देवनागरी)
- Gujarati (ગુજરાતી)

Language Selection Rules:
- Clear English → English reply
- Hindi script → Hindi reply
- Gujarati script → Gujarati reply
- Mixed, Roman Hindi, broken, casual → Hinglish reply
- NEVER reply in any other language
- NEVER mention language detection

2️⃣ Human-like WhatsApp Tone
- Professional but friendly
- Natural, human replies
- Short WhatsApp-style messages
- Light emojis allowed 😊👍 (no overuse)
- NEVER robotic or scripted

3️⃣ Knowledge Usage Rules
- Answer strictly from available information only
- NEVER guess or hallucinate
- NEVER mention internal sources

Forbidden words:
"document", "documents", "dataset", "knowledge base", "training data", "source"

4️⃣ Fallback Rule (CRITICAL)
If exact information is NOT available:
- Politely say information is not available right now
- Offer help with something else
- Do NOT explain why
- Do NOT mention documents or data

Fallback examples:
- Hinglish: "Is topic pe abhi exact info available nahi hai 😊 Aap kuch aur pooch sakte ho."
- Hindi: "Is vishay par abhi jaankari uplabdh nahi hai 😊"
- English: "I don’t have the right information on this yet 😊"
- Gujarati: "આ વિષય પર હાલમાં ચોક્કસ માહિતી ઉપલબ્ધ નથી 😊"

5️⃣ Personalization
- If user's name is known, use it naturally
- Example: "Hi Rahul 😊", "Thanks for reaching out, Ayesha!"

Generate ONLY the system prompt text.
No explanations.
Keep it under 250 words.
                    `.trim(),
                },
                {
                    role: "user",
                    content: `Create a system prompt for a WhatsApp chatbot with this intent:\n"${intent}"`,
                },
            ],
        });

        const systemPrompt = completion.choices[0]?.message?.content?.trim();

        if (!systemPrompt) {
            throw new Error("Failed to generate system prompt");
        }

        // Save / Update in DB
        const { data: existingMappings } = await supabase
            .from("phone_document_mapping")
            .select("id")
            .eq("phone_number", phone_number);

        if (existingMappings && existingMappings.length > 0) {
            await supabase
                .from("phone_document_mapping")
                .update({ intent, system_prompt: systemPrompt })
                .eq("phone_number", phone_number);
        } else {
            await supabase
                .from("phone_document_mapping")
                .insert({
                    phone_number,
                    intent,
                    system_prompt: systemPrompt,
                    file_id: null,
                });
        }

        return NextResponse.json({
            success: true,
            system_prompt: systemPrompt,
            intent,
        });

    } catch (error) {
        console.error("System prompt generation error:", error);
        return NextResponse.json(
            { error: "Failed to generate system prompt" },
            { status: 500 }
        );
    }
}
