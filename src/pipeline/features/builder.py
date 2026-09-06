# src/pipeline/features/builder.py
"""
NOT NEEDED YET. No consumer requires combining technical + fundamental +
alternative features into one matrix -- DataProcessor.process() /
process_multi_asset() already produce the complete feature set the
environments consume, since only technical indicators exist as a real
source right now. Build this once at least two real feature sources
exist and need to be joined; a "builder" over a single source hasn't
earned its keep as an abstraction.
"""