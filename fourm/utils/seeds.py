# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib

def generate_seed(*seeds):
    # Create a hash object using the SHA-256 algorithm
    hash_object = hashlib.sha256()

    # Combine all seeds into a single string
    combined_seeds = ''.join(str(seed) for seed in seeds)

    # Update the hash object with the combined seeds
    hash_object.update(combined_seeds.encode('utf-8'))

    # Get the hexadecimal digest of the hash object
    hex_digest = hash_object.hexdigest()

    # Convert the hexadecimal digest to a 32-bit integer
    seed_int = int(hex_digest, 16) % 2**32
    
    return seed_int