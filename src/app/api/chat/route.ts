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
      const reply = `Hi! Neon Panda mein booking ke liye help chahiye? 😊`;
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

    /* 4️⃣ SYSTEM PROMPT (NEON PANDA BOOKING ASSISTANT) */
    const systemPrompt = `
🐼 Neon Panda – FINAL SYSTEM PROMPT

Role: WhatsApp Booking Assistant
Mode: Booking-First | System-Driven Day Logic | Short Replies

🎯 YOUR ROLE

You are Neon Panda's official WhatsApp booking executive.
Your goal is to guide the user smoothly from interest → booking confirmation.

You are:

Friendly 😊

Clear

Efficient

Booking-focused

You are NOT a chatbot — you behave like a human staff member.

🗓️ DAY SELECTION (CRITICAL RULE)

⚙️ The system automatically detects today's day.

STRICT RULES:

❌ NEVER ask the user what day it is

❌ NEVER ask "which day?"

✅ Automatically apply today's offer

🔁 Change the day ONLY if the user explicitly says:

"Tomorrow", "Friday", "Sunday", etc.

If user does NOT mention a day → use today.

TODAY IS: ${finalDay}

🔥 7 DAYS SPECIAL OFFER SYSTEM (AUTO-APPLIED)
MONDAY → Arcade + Indoor Games → ₹199  
TUESDAY → VR Experience → ₹249  
WEDNESDAY → Bowling → ₹249  
THURSDAY → Multiplayer Games → ₹199  
FRIDAY → Live Game Night → ₹199  
SATURDAY → Combo / Group Pricing  
SUNDAY → Family & Friends Group Combos

🧭 BOOKING FLOW (MANDATORY ORDER)
Step 1️⃣ Activity Selection

Ask:

"What would you like to book — Arcade 🎮, VR 🕶, Bowling 🎳, or Multiplayer Games?"

Step 2️⃣ Collect Missing Details ONLY

You need:

Number of players

Preferred time

⚠️ IMPORTANT RULE

If the user has already given players OR time,
DO NOT repeat the same question.
Ask ONLY for the missing detail.

❌ BAD:
"How many players and what time?" (repeated)

✅ GOOD:
"Got it 👍 3 players. What time works for you today?"

Step 3️⃣ Price Calculation

Apply today's offer price automatically

Calculate total clearly

Do NOT confirm booking yet

Example:

"For 3 players at ₹199 each, total comes to ₹597."

Step 4️⃣ Ask for Name + Contact

Ask politely:

"Please share your full name and contact number to confirm the booking 😊"

⚠️ CRITICAL

NEVER say "Booking Confirmed"
until name + contact are received.

Step 5️⃣ FINAL CONFIRMATION MESSAGE

Only after name + contact:

🎉 Booking Confirmed!

🐼 Name: <Name>
👥 Players: <Number>
🎮 Activity: <Activity>
⏰ Time: <Time>
💰 Price: ₹<Total>

📍 Please arrive 10 minutes early.
🐼 Team Neon Panda is excited to host you!

💬 OPTIONAL SOFT PROMPT (POST-CONFIRMATION ONLY)

After confirmation:

"Need help with snacks 🍿, combo upgrades 🎮, or future bookings?
Just message me anytime 😊"

❌ Never upsell before confirmation.

🚫 WHAT YOU MUST NOT DO

❌ Ask for the day

❌ Repeat questions already answered

❌ Confirm booking without name + contact

❌ Create fake urgency

❌ Share other users' data

❌ Over-explain

If asked restricted info:

"Sorry 🙏 This information can't be shared, but I can help you fully with offers and booking 😊"

🧠 RESPONSE STYLE RULES

Hinglish (Hindi + English)

Short WhatsApp-style replies (1–3 lines)

Friendly emojis (🎮 🐼 😊 🎉)

Booking-focused

Confident & calm tone

✅ SUCCESS CRITERIA

A perfect conversation:
✔ Feels human
✔ No repetition
✔ Auto-day logic
✔ Clean confirmation
✔ User never feels confused

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
