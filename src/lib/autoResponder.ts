import type { ChatCompletionMessageParam } from "groq-sdk/resources/chat/completions";
import { supabase } from "./supabaseClient";
import { embedText } from "./embeddings";
import { retrieveRelevantChunksFromFiles } from "./retrieval";
import { getFilesForPhoneNumber } from "./phoneMapping";
import { sendWhatsAppMessage } from "./whatsappSender";
import { speechToText } from "./speechToText";
import Groq from "groq-sdk";

/* ---------------- GROQ ---------------- */

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY!,
});

/* ---------------- STATE ---------------- */

type ConversationState = {
  stage: "INIT" | "ACTIVITY_SELECTED" | "DETAILS" | "CONFIRM";
  activity: string | null;
  sub_activity: string | null;
  group_size: number | null;
  date: string | null;
  time: string | null;
  pending_fields: string[];
};

const DEFAULT_STATE: ConversationState = {
  stage: "INIT",
  activity: null,
  sub_activity: null,
  group_size: null,
  date: null,
  time: null,
  pending_fields: [],
};

/* ---------------- HELPERS ---------------- */

function cleanUserName(name?: string | null) {
  if (!name) return null;
  return name.replace(/[^\p{L}\p{N}\s]/gu, "").trim();
}

function greetingPrefix(name?: string | null) {
  if (!name) return "";
  return `Hi ${name} 😊 `;
}

function detectReset(text: string) {
  const resetWords = ["restart", "start again", "menu", "reset"];
  return resetWords.some(w => text.includes(w));
}

function normalizePhone(num: string) {
  return num.replace(/\D/g, "");
}

/* -------- STATE DECISION ENGINE ---------- */

function decideNextState(
  prev: ConversationState,
  userText: string
): ConversationState {
  const state: ConversationState = { ...prev };
  const text = userText.toLowerCase();

  if (!state.activity && text.includes("vr")) {
    state.activity = "VR Games";
    state.stage = "ACTIVITY_SELECTED";
    state.pending_fields = ["group_size", "date", "time"];
  }

  if (state.activity === "VR Games" && text.includes("racing")) {
    state.sub_activity = "VR Racing";
  }

  const num = text.match(/\b\d+\b/);
  if (num && !state.group_size) {
    state.group_size = Number(num[0]);
    state.pending_fields = state.pending_fields.filter(f => f !== "group_size");
  }

  if ((text.includes("pm") || text.includes("am")) && !state.time) {
    state.time = userText;
    state.pending_fields = state.pending_fields.filter(f => f !== "time");
  }

  if (
    text.includes("today") ||
    text.includes("saturday") ||
    /\d{2}\/\d{2}\/\d{4}/.test(text)
  ) {
    state.date = userText;
    state.pending_fields = state.pending_fields.filter(f => f !== "date");
  }

  if (state.activity && state.pending_fields.length === 0) {
    state.stage = "CONFIRM";
  }

  return state;
}

/* ---------------- MAIN ---------------- */

export async function generateAutoResponse(
  fromNumber: string,
  toNumber: string,
  messageText: string | null,
  messageId: string,
  mediaUrl?: string,
  senderName?: string
) {
  try {
    console.log("🚀 Auto responder triggered");

    const cleanFrom = normalizePhone(fromNumber);
    const cleanTo = normalizePhone(toNumber);

    /* 1️⃣ FILES */
    const fileIds = await getFilesForPhoneNumber(cleanTo);
    if (fileIds.length === 0) {
      console.warn("⚠️ No documents mapped to phone");
      return { success: false, noDocuments: true };
    }

    /* 2️⃣ PHONE CONFIG */
    const { data: mappings } = await supabase
      .from("phone_document_mapping")
      .select("id, system_prompt, auth_token, origin, conversation_state")
      .eq("phone_number", cleanTo)
      .limit(1);

    if (!mappings?.length) {
      console.error("❌ Phone config missing");
      return { success: false, error: "Phone config missing" };
    }

    const mapping = mappings[0];
    let state: ConversationState =
      mapping.conversation_state || { ...DEFAULT_STATE };

    /* 3️⃣ USER TEXT */
    let finalUserText = messageText?.trim() || "";
    if (!finalUserText && mediaUrl) {
      const transcript = await speechToText(mediaUrl);
      finalUserText = transcript?.text || "";
    }

    if (!finalUserText) {
      return { success: false, error: "Empty message" };
    }

    /* 4️⃣ STATE UPDATE */
    if (detectReset(finalUserText.toLowerCase())) {
      state = { ...DEFAULT_STATE };
    } else {
      state = decideNextState(state, finalUserText);
    }

    await supabase
      .from("phone_document_mapping")
      .update({ conversation_state: state })
      .eq("id", mapping.id);

    /* 5️⃣ RAG */
    const embedding = await embedText(finalUserText);
    const matches = await retrieveRelevantChunksFromFiles(
      embedding,
      fileIds,
      6
    );
    const contextText = matches.map(m => m.chunk).join("\n\n");

    /* 6️⃣ HISTORY */
    const { data: historyRows } = await supabase
      .from("whatsapp_messages")
      .select("content_text, event_type")
      .or(`from_number.eq.${cleanFrom},to_number.eq.${cleanFrom}`)
      .order("received_at", { ascending: true })
      .limit(10);

    const history: ChatCompletionMessageParam[] = (historyRows || [])
      .filter(m => m.content_text)
      .map(m => ({
        role: m.event_type === "MoMessage" ? "user" : "assistant",
        content: String(m.content_text),
      }));

    /* 7️⃣ SYSTEM PROMPT */
    const systemPrompt = `
You are a human-like booking executive.

CURRENT STATE:
- Stage: ${state.stage}
- Activity: ${state.activity || "not selected"}
- Sub Activity: ${state.sub_activity || "not selected"}
- Group size: ${state.group_size || "missing"}
- Date: ${state.date || "missing"}
- Time: ${state.time || "missing"}
- Pending: ${state.pending_fields.join(", ") || "none"}

RULES:
- Ask ONLY pending details
- If CONFIRM → politely confirm booking
- Hinglish only
- Short WhatsApp replies

KNOWLEDGE:
${contextText || "NO_INFORMATION"}
`.trim();

    /* 8️⃣ LLM */
    const completion = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      temperature: 0.2,
      max_tokens: 300,
      messages: [
        { role: "system", content: systemPrompt },
        ...history,
        { role: "user", content: finalUserText },
      ],
    });

    let reply = completion.choices[0]?.message?.content;
    if (!reply) {
      return { success: false, error: "Empty AI response" };
    }

    /* 9️⃣ GREETING */
    const userName = cleanUserName(senderName);
    if (state.stage === "INIT" && history.length === 0 && userName) {
      reply = greetingPrefix(userName) + reply;
    }

    /* 🔟 SEND WHATSAPP (SAFE) */
    console.log("📤 Sending WhatsApp reply...");

    const sendResult = await sendWhatsAppMessage(
      cleanFrom,
      reply,
      mapping.auth_token,
      mapping.origin
    );

    if (!sendResult?.success) {
      console.error("❌ WhatsApp send failed", sendResult);
      return { success: false, error: "WhatsApp send failed" };
    }

    console.log("✅ WhatsApp sent");

    /* 11️⃣ SAVE MESSAGE */
    await supabase.from("whatsapp_messages").insert({
      message_id: `auto_${messageId}_${Date.now()}`,
      channel: "whatsapp",
      from_number: cleanTo,
      to_number: cleanFrom,
      received_at: new Date().toISOString(),
      content_type: "text",
      content_text: reply,
      sender_name: "AI Assistant",
      event_type: "MtMessage",
      is_in_24_window: true,
    });

    return { success: true, response: reply, sent: true };
  } catch (err) {
    console.error("🔥 Auto-response error:", err);
    return { success: false, error: "Auto responder failed" };
  }
}
