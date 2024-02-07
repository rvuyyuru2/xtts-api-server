# import time
# import torch
# import numpy as np
# import torchaudio
# import logging

# logger = logging.getLogger(__name__)

# async def stream_generation(self, text, speaker_name, speaker_wav, language, output_file):
#     generate_start_time = time.time()  # Log the start time of the operation

#     # Retrieve or create necessary latent vectors for the model
#     gpt_cond_latent, speaker_embedding = self.get_or_create_latents(speaker_name, speaker_wav)

#     # Initialize an empty list to hold the tensor chunks for final audio file creation
#     file_chunks = []

#     # Stream the generation process, yielding audio chunks as they're produced
#     try:
#         for chunk in self.model.inference_stream(text, language, speaker_embedding=speaker_embedding, gpt_cond_latent=gpt_cond_latent, **self.tts_settings, stream_chunk_size=self.stream_chunk_size):
            
#             # Handle chunk aggregation if the model outputs a list of tensors
#             if isinstance(chunk, list):
#                 chunk = torch.cat(chunk, dim=0)
#             file_chunks.append(chunk)
            
#             # Prepare the chunk for streaming by converting it to bytes
#             chunk = chunk.cpu().numpy()
#             chunk = np.clip(chunk, -1, 1)  # Ensure the values are within the audio range
#             chunk = (chunk * 32767).astype(np.int16)  # Convert to 16-bit PCM format
#             yield chunk.tobytes()  # Yield the audio chunk as bytes for streaming

#         # After all chunks are processed, concatenate them and save the final audio file
#         if file_chunks:
#             wav = torch.cat(file_chunks, dim=0)
#             torchaudio.save(output_file, wav.cpu().squeeze().unsqueeze(0), 24000)
#         else:
#             logger.warning("No audio generated.")
#     except Exception as e:
#         logger.error(f"An error occurred during stream generation: {e}")
#     finally:
#         generate_end_time = time.time()  # Log the end time of the operation
#         generate_elapsed_time = generate_end_time - generate_start_time  # Calculate the elapsed time
#         logger.info(f"Processing time: {generate_elapsed_time:.2f} seconds.")  # Log the processing time

