"""T-Head PPU backend extension ops.

PPU-ZW810E is exposed through the torch.cuda compatibility interface by the
T-Head runtime. No vendor-specific Python op replacements are needed for the
initial elementwise smoke path.
"""
