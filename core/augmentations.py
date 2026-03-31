import numpy as np

def generate_d4_augmentations(grid):
    """
    Generates all 8 permutations of the D4 Symmetry Group for a given 2D ARC grid.
    These synthetic spatial variations prevent the SNN from over-fitting specific pixel
    coordinates and enforce 'translation/rotation-invariant' Induction learning.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
        
    # Standard identity
    augs = [grid]
    
    # 3 Rotations (90, 180, 270 degrees)
    augs.append(np.rot90(grid, k=1))
    augs.append(np.rot90(grid, k=2))
    augs.append(np.rot90(grid, k=3))
    
    # 4 Flips (Horizontal, Vertical, Main Diagonal, Anti-Diagonal)
    flipped_h = np.fliplr(grid)
    augs.append(flipped_h)
    augs.append(np.flipud(grid))
    
    # Diagonals are equivalent to transpose of the horizontal flip variations
    augs.append(np.transpose(grid))
    augs.append(np.transpose(np.rot90(grid, k=2)))
    
    return augs

def augment_arc_pairs(train_pairs):
    """
    Reads standard train_pairs (each containing 'input' and 'output') and returns an
    expanded batch containing 8x the data geometry by systematically morphing the pairs.
    """
    augmented_pairs = []
    
    for pair in train_pairs:
        inputs_aug = generate_d4_augmentations(pair['input'])
        outputs_aug = generate_d4_augmentations(pair['output'])
        
        for i_grid, o_grid in zip(inputs_aug, outputs_aug):
            augmented_pairs.append({'input': i_grid, 'output': o_grid})
            
    return augmented_pairs
