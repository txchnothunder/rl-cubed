"""
base_cube.py

Defines a base class for Rubik's Cubes of any size.
Handles initialization and colored display.
"""

class BaseCube:
    COLORS = {
        'W': '\033[97m',        # White
        'R': '\033[91m',        # Red
        'G': '\033[92m',        # Green
        'Y': '\033[93m',        # Yellow
        'O': '\033[38;5;208m',  # Orange
        'B': '\033[94m',        # Blue
        'END': '\033[0m'        # Reset
    }

    def __init__(self, size):
        self.size = size  # 2 or 3
        self._cube = self.init_cube()

    def init_cube(self):
        """Initialize a solved cube as a single string."""
        colors = ['W', 'R', 'G', 'Y', 'O', 'B']
        return ''.join([c * (self.size**2) for c in colors])

    def _rotate_face_in_list(self, cube_list, face_key, direction):
        """
        Rotate a face (U, D, F, B, L, R) 90 degrees in-place on cube_list.
        direction = 1 (clockwise when looking at the face),
                    -1 (counterclockwise)
        """
        faces = {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}
        step = self.size ** 2
        start = faces[face_key] * step

        face = cube_list[start:start + step]
        new_face = face[:]  # copy

        for r in range(self.size):
            for c in range(self.size):
                old_i = r * self.size + c
                if direction == 1:  # CW
                    new_r = c
                    new_c = self.size - 1 - r
                else:  # CCW
                    new_r = self.size - 1 - c
                    new_c = r
                new_i = new_r * self.size + new_c
                new_face[new_i] = face[old_i]

        cube_list[start:start + step] = new_face
    
    def turn_right_90(self, row):
        """
        Rotates the given row of the cube to the right by 90 degrees.
        Row numbering is 1-based: 1 = top row, 2 = second row, etc.
        Negative values rotate LEFT instead of RIGHT.
        
        For a horizontal rotation:
        - Positive: Front → Right → Back → Left → Front
        - Negative: Front → Left → Back → Right → Front
        """
        if row == 0:
            raise ValueError("Row must be non-zero (1-based indexing)")
        
        direction = 1 if row > 0 else -1  # 1 = right, -1 = left
        row_index = abs(row) - 1  # 0-based

        if row_index >= self.size:
            raise ValueError(f"Invalid row: must be between 1 and {self.size} (or negative)")

        faces = {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}
        step = self.size ** 2

        # Calculate the starting position of this row in each side face
        f_start = faces['F'] * step + row_index * self.size
        r_start = faces['R'] * step + row_index * self.size
        b_start = faces['B'] * step + row_index * self.size
        l_start = faces['L'] * step + row_index * self.size

        # Extract the row from each face (from the original cube!)
        f_row = self._cube[f_start:f_start + self.size]
        r_row = self._cube[r_start:r_start + self.size]
        b_row = self._cube[b_start:b_start + self.size]
        l_row = self._cube[l_start:l_start + self.size]

        # Work on a mutable copy
        new_cube = list(self._cube)
        
        if direction == 1:  # Right rotation: F → R → B → L → F
            for i in range(self.size):
                new_cube[r_start + i] = f_row[i]
                new_cube[b_start + i] = r_row[i]
                new_cube[l_start + i] = b_row[i]
                new_cube[f_start + i] = l_row[i]
        else:  # Left rotation: F → L → B → R → F
            for i in range(self.size):
                new_cube[l_start + i] = f_row[i]
                new_cube[b_start + i] = l_row[i]
                new_cube[r_start + i] = b_row[i]
                new_cube[f_start + i] = r_row[i]

        # 🔁 Now rotate U/D faces IN THE SAME new_cube list

        if row_index == 0:
            # Top row => rotate U face
            self._rotate_face_in_list(new_cube, 'U', direction)
        elif row_index == self.size - 1:
            # Bottom row => rotate D face
            # Note: bottom face orientation is flipped relative to the horizontal turn.
            self._rotate_face_in_list(new_cube, 'D', -direction)

        # Commit all changes at once
        self._cube = ''.join(new_cube)


    def turn_left_90(self, row):
        self.turn_right_90(-1*row)

    def turn_upward_90(self, col):
        """
        Rotates the given column of the cube upward by 90 degrees.
        Column numbering is 1-based: 1 = leftmost column, 2 = second column, etc.
        Negative values rotate DOWNWARD instead of UPWARD.
        
        For a vertical rotation:
        - Positive (upward): Front → Up → Back → Down → Front
        - Negative (downward): Front → Down → Back → Up → Front
        """
        # Determine direction
        if col == 0:
            raise ValueError("Column must be non-zero (1-based indexing)")
        
        direction = 1 if col > 0 else -1  # 1 = upward, -1 = downward
        col_index = abs(col) - 1  # Convert to 0-based
        
        if col_index >= self.size:
            raise ValueError(f"Invalid column: must be between 1 and {self.size} (or negative)")

        faces = {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}
        step = self.size ** 2

        # Extract columns from each face
        f_col = [self._cube[faces['F'] * step + row * self.size + col_index] 
                for row in range(self.size)]
        
        u_col = [self._cube[faces['U'] * step + row * self.size + col_index] 
                for row in range(self.size)]
        
        b_col = [self._cube[faces['B'] * step + row * self.size + (self.size - 1 - col_index)] 
                for row in range(self.size)]
        
        d_col = [self._cube[faces['D'] * step + row * self.size + col_index] 
                for row in range(self.size)]

        new_cube = list(self._cube)
        
        if direction == 1:  # Upward: F → U → B → D → F
            for row in range(self.size):
                new_cube[faces['U'] * step + row * self.size + col_index] = f_col[row]
                new_cube[faces['B'] * step + row * self.size + (self.size - 1 - col_index)] = u_col[row]
                new_cube[faces['D'] * step + row * self.size + col_index] = b_col[row]
                new_cube[faces['F'] * step + row * self.size + col_index] = d_col[row]
        else:  # Downward: F → D → B → U → F
            for row in range(self.size):
                new_cube[faces['D'] * step + row * self.size + col_index] = f_col[row]
                new_cube[faces['B'] * step + row * self.size + (self.size - 1 - col_index)] = d_col[row]
                new_cube[faces['U'] * step + row * self.size + col_index] = b_col[row]
                new_cube[faces['F'] * step + row * self.size + col_index] = u_col[row]

        # Rotate left or right face if needed
        if col_index == 0:
            # left face rotates opposite of upward direction
            self._rotate_face_in_list(new_cube, 'L', -direction)
        elif col_index == self.size - 1:
            # right face rotates same direction
            self._rotate_face_in_list(new_cube, 'R', direction)

        # commit all changes
        self._cube = ''.join(new_cube)

    def turn_downward_90(self, row):
        self.turn_upward_90(-1 * row)
    
    def turn_clockwise_90(self, slice_num):
        """
        Rotates the given depth slice clockwise by 90 degrees (when viewed from the right side).
        Slice numbering is 1-based: 1 = front slice, 2 = second slice, etc.
        Negative values rotate COUNTER-CLOCKWISE instead of CLOCKWISE.
        
        For a depth rotation:
        - Positive (clockwise): Top → Right → Bottom → Left → Top
        - Negative (counter-clockwise): Top → Left → Bottom → Right → Top
        """
        # Determine direction
        if slice_num == 0:
            raise ValueError("Slice must be non-zero (1-based indexing)")
        
        direction = 1 if slice_num > 0 else -1  # 1 = clockwise, -1 = counter-clockwise
        slice_index = abs(slice_num) - 1  # Convert to 0-based
        
        if slice_index >= self.size:
            raise ValueError(f"Invalid slice: must be between 1 and {self.size} (or negative)")

        faces = {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}
        step = self.size ** 2

        # Extract the slice from each face
        # Top: row at slice_index from the back (reading left to right)
        u_slice = [self._cube[faces['U'] * step + slice_index * self.size + col] 
                for col in range(self.size)]
        
        # Right: column at slice_index from the left (reading top to bottom)
        r_slice = [self._cube[faces['R'] * step + row * self.size + slice_index] 
                for row in range(self.size)]
        
        # Bottom: row at slice_index from the front (reading left to right)
        d_slice = [self._cube[faces['D'] * step + slice_index * self.size + col] 
                for col in range(self.size)]
        
        # Left: column at (size - 1 - slice_index) from the right (reading top to bottom)
        l_slice = [self._cube[faces['L'] * step + row * self.size + (self.size - 1 - slice_index)] 
                for row in range(self.size)]

        new_cube = list(self._cube)
        
        if direction == 1:  # Clockwise: U → R → D → L → U
            for i in range(self.size):
                new_cube[faces['R'] * step + i * self.size + slice_index] = u_slice[i]
                new_cube[faces['D'] * step + slice_index * self.size + i] = r_slice[i]
                new_cube[faces['L'] * step + i * self.size + (self.size - 1 - slice_index)] = d_slice[i]
                new_cube[faces['U'] * step + slice_index * self.size + i] = l_slice[i]
        else:  # Counter-clockwise: U → L → D → R → U
            for i in range(self.size):
                new_cube[faces['L'] * step + i * self.size + (self.size - 1 - slice_index)] = u_slice[i]
                new_cube[faces['D'] * step + slice_index * self.size + i] = l_slice[i]
                new_cube[faces['R'] * step + i * self.size + slice_index] = d_slice[i]
                new_cube[faces['U'] * step + slice_index * self.size + i] = r_slice[i]

        self._cube = ''.join(new_cube)

    def turn_counterclockwise_90(self, slice_num):
        self.turn_clockwise_90(-1 * slice_num)

    def turn_right(self, row, degree):
        """Turns a row of the cube by a degree in multiples of 90. If it is not an exact multiple of 90, it will fail."""
        if degree % 90 != 0:
            raise ValueError("Degree must be a multiple of 90")
        
        times = (degree // 90) % 4  # Modulo 4 because 4 rotations = full circle
        for _ in range(times):
            self.turn_right_90(row)

    def turn_left(self, row, degree):
        """Turns a row of the cube left by a degree in multiples of 90. If it is not an exact multiple of 90, it will fail."""
        if degree % 90 != 0:
            raise ValueError("Degree must be a multiple of 90")
        
        times = (degree // 90) % 4
        for _ in range(times):
            self.turn_right_90(-row)  # Negative row = left rotation

    def turn_upward(self, col, degree):
        """Turns a column of the cube upward by a degree in multiples of 90. If it is not an exact multiple of 90, it will fail."""
        if degree % 90 != 0:
            raise ValueError("Degree must be a multiple of 90")
        
        times = (degree // 90) % 4
        for _ in range(times):
            self.turn_upward_90(col)

    def turn_downward(self, col, degree):
        """Turns a column of the cube downward by a degree in multiples of 90. If it is not an exact multiple of 90, it will fail."""
        if degree % 90 != 0:
            raise ValueError("Degree must be a multiple of 90")
        
        times = (degree // 90) % 4
        for _ in range(times):
            self.turn_upward_90(-col)  # Negative col = downward rotation

    def turn_clockwise(self, slice_num, degree):
        """Turns a depth slice clockwise by a degree in multiples of 90. If it is not an exact multiple of 90, it will fail."""
        if degree % 90 != 0:
            raise ValueError("Degree must be a multiple of 90")
        
        times = (degree // 90) % 4
        for _ in range(times):
            self.turn_clockwise_90(slice_num)

    def turn_counterclockwise(self, slice_num, degree):
        """Turns a depth slice counter-clockwise by a degree in multiples of 90. If it is not an exact multiple of 90, it will fail."""
        if degree % 90 != 0:
            raise ValueError("Degree must be a multiple of 90")
        
        times = (degree // 90) % 4
        for _ in range(times):
            self.turn_clockwise_90(-slice_num)  # Negative slice = counter-clockwise rotation

    def colorize(self, sticker):
            return f"{self.COLORS[sticker]}{sticker}{self.COLORS['END']}"
        
    def print_internal(self):
        """Return a colored string representation of the cube.
        Appears as how the computer interprets the string."""
        colored = ''.join(self.colorize(sticker) for sticker in self._cube)
        print(colored)


    def __str__(self):
        """Return a colored string representation of the cube in a cube net layout."""
        s = ""
        size = self.size
        c = self._cube

        # Map faces to indices
        faces = {
            'U': 0,
            'R': 1,
            'F': 2,
            'D': 3,
            'L': 4,
            'B': 5
        }
        step = size**2

        # Helper to get a face row
        def get_face_row(face, row):
            start = faces[face]*step + row*size
            end = start + size
            return [self.colorize(x) for x in c[start:end]]

        # Print top (U)
        for row in range(size):
            s += ' ' * (size*2) + ' '.join(get_face_row('U', row)) + '\n'

        # Print middle: L, F, R, B
        for row in range(size):
            s += ' '.join(get_face_row('L', row)) + ' '
            s += ' '.join(get_face_row('F', row)) + ' '
            s += ' '.join(get_face_row('R', row)) + ' '
            s += ' '.join(get_face_row('B', row)) + '\n'

        # Print bottom (D)
        for row in range(size):
            s += ' ' * (size*2) + ' '.join(get_face_row('D', row)) + '\n'

        return s

