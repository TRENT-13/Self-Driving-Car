import os
import sys

# Debug environment
print(f"DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")
print(f"PYGLET_HEADLESS: {os.environ.get('PYGLET_HEADLESS', 'NOT SET')}")

# Test pyglet first
print("Testing pyglet window creation...")
try:
    import pyglet
    print(f"Pyglet version: {pyglet.version}")
    
    # Create a simple test window
    test_window = pyglet.window.Window(width=400, height=300, caption="Test Window")
    print("✅ Test window created successfully!")
    
    @test_window.event
    def on_draw():
        test_window.clear()
    
    # Auto-close after 2 seconds
    def close_window(dt):
        test_window.close()
    
    pyglet.clock.schedule_once(close_window, 2.0)
    pyglet.app.run()
    
    print("Test window closed automatically")
    
except Exception as e:
    print(f"❌ Pyglet test failed: {e}")
    import traceback
    traceback.print_exc()

print("Now testing Duckietown...")

try:
    import gym_duckietown
    from gym_duckietown.simulator import Simulator
    
    print("Creating Duckietown simulator...")
    env = Simulator(
        seed=123,
        map_name="regress_4way_adam",
        max_steps=500001,
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4,
        full_transparency=True,
        distortion=True,
    )
    
    print("✅ Simulator created successfully!")
    print("Starting simulation loop...")
    
    steps = 0
    while True:  # Limit steps for testing
        action = [0.1, 0.1]
        observation, reward, done, misc = env.step(action)
        env.render()
        steps += 1
        
        if steps % 10 == 0:
            print(f"Step {steps}")
        
        if done:
            print("Episode done, resetting...")
            env.reset()
    
    print("✅ Simulation completed successfully!")
    
except Exception as e:
    print(f"❌ Duckietown simulation failed: {e}")
    import traceback
    traceback.print_exc()
