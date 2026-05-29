from src.cube3x3 import Cube3x3
from src.cube2x2 import Cube2x2

def cube2x2_test():
    print("Testing 2x2 Cube:")
    cube = Cube2x2()
    print(cube)

    cube.turn_right_90(1)
    print(cube)

    cube.turn_left_90(1)
    print(cube)

    cube.turn_upward_90(2)
    print(cube)

    cube.turn_downward_90(2)
    print(cube)

    cube.turn_clockwise_90(1)
    print(cube)

    cube.turn_counterclockwise_90(1)
    print(cube)

def cube3x3_test():
    print("Testing 3x3 Cube:")
    cube = Cube3x3()
    print(cube)

    cube.turn_downward_90(3)
    print(cube)
    
    cube.turn_left_90(3)
    print(cube)

    cube.turn_upward_90(3)
    print(cube)

    cube.turn_right_90(3)
    print(cube)

def main():
    cube2x2_test()
    cube3x3_test()

if __name__ == "__main__":
    main()