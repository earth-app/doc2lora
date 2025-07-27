/**
 * Cloudflare Worker using doc2lora generated adapter
 *
 * This Worker demonstrates how to use a custom LoRA adapter
 * trained on software developer documentation.
 */

export default {
  async fetch(request, env) {
    // Handle CORS
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }

    try {
      // Only handle POST requests
      if (request.method !== 'POST') {
        return new Response('Method not allowed', {
          status: 405,
          headers: { 'Access-Control-Allow-Origin': '*' }
        });
      }

      const { message } = await request.json();

      if (!message) {
        return new Response('Message is required', {
          status: 400,
          headers: { 'Access-Control-Allow-Origin': '*' }
        });
      }

      // Use Cloudflare AI with our custom LoRA adapter
      const ai = new Ai(env.AI);

      const response = await ai.run('@cf/mistralai/mistral-7b-instruct-v0.2-lora', {
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant that answers questions about software development based on the provided training data. Be specific and detailed in your responses."
          },
          {
            role: "user",
            content: message
          }
        ],
        lora: "software_dev_adapter", // Reference to our uploaded adapter
        max_tokens: 512,
        temperature: 0.7
      });

      return new Response(JSON.stringify({
        success: true,
        response: response.response,
        model: '@cf/mistralai/mistral-7b-instruct-v0.2-lora',
        adapter: 'software_dev_adapter'
      }), {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      });

    } catch (error) {
      console.error('Error:', error);

      return new Response(JSON.stringify({
        success: false,
        error: error.message,
        details: 'Check that your LoRA adapter is properly uploaded and configured'
      }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      });
    }
  }
};
