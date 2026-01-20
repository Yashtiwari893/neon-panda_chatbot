import { NextResponse } from "next/server";
import Groq from "groq-sdk";
import { supabase } from "@/lib/supabaseClient";
import { embedText } from "@/lib/embeddings";
import { retrieveRelevantChunks } from "@/lib/retrieval";

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY!,
});

const SMALL_TALK = ["hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "bye"];

function isSmallTalk(message: string) {
  return SMALL_TALK.includes(message.trim().toLowerCase());
}

function getSystemDay() {
  return new Date().toLocaleDateString("en-US", { weekday: "long", timeZone: "Asia/Kolkata" });
}

function detectExplicitDay(message: string): string | null {
  const lower = message.toLowerCase();
  const days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
  for (const day of days) {
    if (lower.includes(day)) {
      return day.charAt(0).toUpperCase() + day.slice(1);
    }
  }
  if (lower.includes("tomorrow") || lower.includes("kal")) {
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);
    return tomorrow.toLocaleDateString("en-US", { weekday: "long", timeZone: "Asia/Kolkata" });
  }
  if (lower.includes("aaj")) {
    return getSystemDay();
  }
  return null;
}

export async function POST(req: Request) {
  try {
    const { session_id, message, file_id } = await req.json();

    if (!session_id || !message) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    const systemDay = getSystemDay();
    const explicitDay = detectExplicitDay(message);
    const finalDay = explicitDay || systemDay;

    /* 1️⃣ Handle small talk WITHOUT embeddings */
    if (isSmallTalk(message)) {
      const reply = `Hi 😊 Kaise help kar sakta hoon?`;
      return new Response(reply, { status: 200 });
    }

    /* 2️⃣ Try embeddings safely */
    let contextText = "";
    try {
      const embedding = await embedText(message);
      if (embedding) {
        const matches = await retrieveRelevantChunks(embedding, file_id, 8);
        contextText = matches.map(m => m.chunk).join("\n\n");
      }
    } catch (err) {
      console.warn("⚠️ Embedding failed, continuing without RAG");
    }

    /* 3️⃣ Load history */
    const { data: historyRows } = await supabase
      .from("messages")
      .select("role, content")
      .eq("session_id", session_id)
      .order("created_at", { ascending: true });

    const history = (historyRows || []).map(m => ({
      role: m.role,
      content: m.content,
    }));

    /* 4️⃣ SYSTEM PROMPT (DAY AWARE + 4 LANGUAGES) */
    const systemPrompt = `
You are a WhatsApp chatbot.

TODAY IS: ${finalDay}

LANGUAGE:
Reply ONLY in:
- Hinglish
- English
- Hindi
- Gujarati

RULES:
- Match user's language
- Friendly & natural
- Short replies
- Light emojis 😊
- NEVER ask user what day it is
- Always respond using detected system day

INTELLIGENCE:
- Understand intent (offer / discount / deal)
- Use ONLY info below
- Select ONLY TODAY's relevant content
- Ignore other days

FALLBACK:
If info missing:
"Is topic pe abhi exact info available nahi hai 😊"

INFO:
${contextText || "NO_INFORMATION_AVAILABLE"}
`.trim();

    const completion = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        { role: "system", content: systemPrompt },
        ...history,
        { role: "user", content: message },
      ],
      temperature: 0.3,
    });

    const answer = completion.choices[0]?.message?.content || 
      "Abhi ispe exact info available nahi hai 😊";

    return new Response(answer, { status: 200 });

  } catch (error) {
    console.error("CHAT_ERROR:", error);
    return new Response(
      "Thoda sa issue aa gaya 😅 Please thodi der baad try karein.",
      { status: 200 }
    );
  }
}

// Auto day selection logic applied, user prompt removed.
