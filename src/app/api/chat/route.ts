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
  
  if (lower.includes("aaj") || lower.includes("today")) {
    return getSystemDay();
  }
  
  return null;
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { session_id, message, file_id } = body;

    if (!session_id || !message) {
      return NextResponse.json(
        { error: "session_id and message are required" },
        { status: 400 }
      );
    }

    const systemDay = getSystemDay();
    const explicitDay = detectExplicitDay(message);
    const finalDay = explicitDay || systemDay;

    // Handle small talk WITHOUT embeddings
    if (isSmallTalk(message)) {
      const reply = `Hi! Neon Panda mein booking ke liye help chahiye? 😊`;
      
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(reply));
          controller.close();
        }
      });

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
        }
      });
    }

    // Embed the user query and retrieve relevant chunks
    let contextText = "";
    try {
      const queryEmbedding = await embedText(message);
      
      if (queryEmbedding) {
        const matches = await retrieveRelevantChunks(queryEmbedding, file_id, 8);
        contextText = matches.map((m) => m.chunk).join("\n\n");
      }
    } catch (err) {
      console.warn("⚠️ Embedding failed, continuing without RAG context");
    }

    // Load conversation history
    const { data: historyRows } = await supabase
      .from("messages")
      .select("role, content")
      .eq("session_id", session_id)
      .order("created_at", { ascending: true });

    const history = (historyRows || []).map(m => ({
      role: m.role,
      content: m.content
    }));

    // Build system prompt combining both approaches
    const systemPrompt = `
You are Neon Panda's WhatsApp booking executive. Be friendly, human-like, booking-focused.

TODAY IS: ${finalDay}

DAILY OFFERS:
MONDAY → Arcade + Indoor Games → ₹199  
TUESDAY → VR Experience → ₹249  
WEDNESDAY → Bowling → ₹249  
THURSDAY → Multiplayer Games → ₹199  
FRIDAY → Live Game Night → ₹199  
SATURDAY → Combo / Group Pricing  
SUNDAY → Family & Friends Group Combos

${contextText ? `\nDOCUMENT CONTEXT:\n${contextText}\n` : ''}

BOOKING FLOW:
1. Ask activity: "Arcade 🎮, VR 🕶, Bowling 🎳, or Multiplayer?"
2. Ask missing details: players, time (one question at a time, no repeats)
3. Calculate price using today's offer
4. Ask name + contact
5. Confirm booking with format

STRICT RULES:
- ONLY use information from DOCUMENT CONTEXT if provided
- If answer not in context, use your booking knowledge
- NEVER ask for day (auto-detected)
- Reply in user's language (Hinglish preferred)
- Short replies (1-3 lines max)
- Friendly emojis
- No upselling before confirmation
- Remember context, don't repeat questions
- NEVER offer to do tasks you cannot do (generate QR codes, create files, etc.)

CONFIRMATION FORMAT:
🎉 Booking Confirmed!

🐼 Name: <Name>
👥 Players: <Number>
🎮 Activity: <Activity>
⏰ Time: <Time>
💰 Price: ₹<Total>

📍 Please arrive 10 minutes early.
🐼 Team Neon Panda is excited to host you!

FALLBACK: "Sorry, I don't have that info right now."
`.trim();

    const messages = [
      { role: "system", content: systemPrompt },
      ...history,
      { role: "user", content: message }
    ];

    // Call Groq with streaming
    const completion = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages,
      temperature: 0.3,
      stream: true
    });

    // Create streaming response
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content || "";
            if (content) {
              controller.enqueue(encoder.encode(content));
            }
          }
          controller.close();
        } catch (error) {
          console.error("Streaming error:", error);
          controller.error(error);
        }
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked'
      }
    });

  } catch (err: unknown) {
    console.error("CHAT_ERROR:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
