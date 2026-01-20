import { supabase } from "./supabaseClient";
import { embedText } from "./embeddings";
import { retrieveRelevantChunksFromFiles } from "./retrieval";
import { getFilesForPhoneNumber } from "./phoneMapping";
import { sendWhatsAppMessage } from "./whatsappSender";
import Groq from "groq-sdk";

/* ---------------- GROQ ---------------- */

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY!,
});

/* ---------------- TYPES ---------------- */

export type AutoResponseResult = {
    success: boolean;
    response?: string;
    error?: string;
    noDocuments?: boolean;
    sent?: boolean;
};

/* ---------------- HELPERS ---------------- */

function normalizePhone(num: string): string {
    return num.replace(/\D/g, "");
}

function safeString(val: unknown): string {
    return typeof val === "string" ? val : "";
}

/* ---------------- MAIN ---------------- */

export async function generateAutoResponse(
    fromNumber: string,
    toNumber: string,
    messageText: string,
    messageId: string
): Promise<AutoResponseResult> {
    try {
        console.log("🚀 Auto responder triggered");

        const cleanFrom = normalizePhone(fromNumber);
        const cleanTo = normalizePhone(toNumber);

        console.log("🔎 Phone lookup", { cleanFrom, cleanTo });

        /* 1️⃣ FILES */
        const fileIds = await getFilesForPhoneNumber(cleanTo);

        if (!fileIds || fileIds.length === 0) {
            console.warn("⚠️ No documents mapped to phone:", cleanTo);
            return { success: false, noDocuments: true };
        }

        /* 2️⃣ PHONE CONFIG */
        const { data: phoneMappings, error: mappingError } = await supabase
            .from("phone_document_mapping")
            .select("system_prompt, intent, auth_token, origin")
            .eq("phone_number", cleanTo)
            .limit(1);

        if (mappingError || !phoneMappings || phoneMappings.length === 0) {
            console.error("❌ Phone config missing", mappingError);
            return { success: false, error: "Phone config missing" };
        }

        const mapping = phoneMappings[0];

        const systemPromptBase = safeString(mapping.system_prompt);
        const auth_token = safeString(mapping.auth_token);
        const origin = safeString(mapping.origin);

        if (!auth_token || !origin) {
            return {
                success: false,
                error: "WhatsApp API credentials missing for this phone number",
            };
        }

        /* 3️⃣ USER MESSAGE */
        const userText = safeString(messageText).trim();
        if (!userText) {
            return { success: false, error: "Empty message" };
        }

        /* 4️⃣ EMBEDDING */
        const embedding = await embedText(userText);
        if (!embedding) {
            return { success: false, error: "Embedding generation failed" };
        }

        /* 5️⃣ RAG */
        const matches = await retrieveRelevantChunksFromFiles(
            embedding,
            fileIds,
            5
        );

        const contextText =
            matches && matches.length > 0
                ? matches.map(m => m.chunk).join("\n\n")
                : "NO_RELEVANT_INFORMATION";

        /* 6️⃣ HISTORY */
        const { data: historyRows } = await supabase
            .from("whatsapp_messages")
            .select("content_text, event_type")
            .or(`from_number.eq.${cleanFrom},to_number.eq.${cleanFrom}`)
            .order("received_at", { ascending: true })
            .limit(20);

        const history =
            historyRows?.filter(
                m =>
                    typeof m.content_text === "string" &&
                    (m.event_type === "MoMessage" || m.event_type === "MtMessage")
            ).map(m => ({
                role: m.event_type === "MoMessage" ? "user" as const : "assistant" as const,
                content: m.content_text,
            })) ?? [];

        /* 7️⃣ SYSTEM PROMPT */
        const documentRules = `
You must ONLY answer using the document context.

STRICT RULES:
- Use ONLY the CONTEXT below
- If answer not found, say: "I don't have that information in the document"
- No assumptions
- No external knowledge
- Short WhatsApp-style replies
- Max 5 lines
`.trim();

        const systemPrompt = systemPromptBase
            ? `${systemPromptBase}\n\n${documentRules}`
            : `You are a helpful WhatsApp assistant.\n\n${documentRules}`;

        const messages = [
            {
                role: "system" as const,
                content: `${systemPrompt}\n\nCONTEXT:\n${contextText}`,
            },
            ...history.slice(-10),
            { role: "user" as const, content: userText },
        ];

        /* 8️⃣ LLM */
        const completion = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            messages,
            temperature: 0.2,
            max_tokens: 300,
        });

        const reply = completion.choices[0]?.message?.content?.trim();
        if (!reply) {
            return { success: false, error: "Empty AI response" };
        }

        /* 9️⃣ SEND WHATSAPP */
        const sendResult = await sendWhatsAppMessage(
            cleanFrom,
            reply,
            auth_token,
            origin
        );

        if (!sendResult.success) {
            console.error("❌ WhatsApp send failed:", sendResult.error);
            return {
                success: false,
                response: reply,
                sent: false,
                error: "WhatsApp send failed",
            };
        }

        /* 🔟 SAVE BOT MESSAGE */
        const botMessageId = `auto_${messageId}_${Date.now()}`;

        await supabase.from("whatsapp_messages").insert({
            message_id: botMessageId,
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

        /* 11️⃣ MARK ORIGINAL AS RESPONDED */
        await supabase
            .from("whatsapp_messages")
            .update({
                is_responded: true,
                response_sent_at: new Date().toISOString(),
            })
            .eq("message_id", messageId);

        console.log("✅ Auto-response sent successfully");

        return {
            success: true,
            response: reply,
            sent: true,
        };
    } catch (err) {
        console.error("🔥 Auto-response error:", err);
        return {
            success: false,
            error: err instanceof Error ? err.message : "Unknown error",
        };
    }
}
