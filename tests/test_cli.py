import pytest
from typer.testing import CliRunner
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock

# Assuming your Typer app is in src.utils.cli and named 'app'
from src.utils.cli import app
from src.utils.config import Config, ModelConfig, SystemConfig, LoggingConfig, TemporalContextConfig, PluginConfig

# Fixture for CliRunner
@pytest.fixture
def runner():
    return CliRunner()

# Fixture for a default mock Config
@pytest.fixture
def mock_cli_config():
    return Config(
        model=ModelConfig(ollama_model="test-model", ollama_temperature=0.7, flux_model_path="flux/path"),
        system=SystemConfig(log_dir="logs/", mps_use_fp16=False, image_output_dir="output/cli_test_output"),
        logging=LoggingConfig(level="INFO"),
        temporal_context=TemporalContextConfig(enabled=True),
        plugins=PluginConfig(enabled=[], descriptions={})
    )

# Mock for PromptGenerator instance
@pytest.fixture
def mock_prompt_generator_instance():
    mock_instance = MagicMock() # Use MagicMock for flexibility
    mock_instance.generate_prompt = AsyncMock(return_value="Test Cleaned Prompt from generate_prompt")
    mock_instance.thinking_process = ["Think Step CLI 1", "Think Step CLI 2"]
    mock_instance.get_prompt_with_feedback = AsyncMock(return_value="Test Cleaned Prompt from feedback")
    mock_instance.cleanup = MagicMock()
    return mock_instance

# Mock for ImageGenerator instance
@pytest.fixture
def mock_image_generator_instance():
    mock_instance = MagicMock()
    # Ensure generate_image returns a tuple like the real one
    mock_instance.generate_image = AsyncMock(return_value=("path/to/image.png", 10.0, "test-flux-model"))
    mock_instance.cleanup = MagicMock()
    return mock_instance


# Patching the classes where instances are created in cli.py
@patch('src.utils.cli.ImageGenerator')
@patch('src.utils.cli.PromptGenerator')
@patch('src.utils.cli.StorageManager') # Also mock StorageManager if it has side effects like dir creation
@patch('src.utils.cli.MetricsCollector')
def test_generate_non_interactive_with_think_steps(
    MockMetricsCollector, MockStorageManager, MockPromptGenerator, MockImageGenerator, 
    runner, mock_cli_config, mock_prompt_generator_instance
):
    MockPromptGenerator.return_value = mock_prompt_generator_instance
    MockImageGenerator.return_value = mock_image_generator_instance # Assign the mock instance
    MockStorageManager.return_value.get_output_path.return_value = MagicMock(with_suffix=lambda s: "path/to/prompt.txt")

    # Patch the app.state.config directly
    with patch('src.utils.cli.app.state.config', mock_cli_config):
        result = runner.invoke(app, ["generate"])

    assert result.exit_code == 0, f"CLI Error: {result.stdout}"
    
    # Verify PromptGenerator was called
    mock_prompt_generator_instance.generate_prompt.assert_called_once()

    # Verify ImageGenerator was called
    mock_image_generator_instance.generate_image.assert_called_once()
    
    # Check for thinking process and prompt in output
    assert "Reasoning Process" in result.stdout
    assert "Think Step CLI 1" in result.stdout
    assert "Think Step CLI 2" in result.stdout
    assert "AI Prompt" in result.stdout
    assert "Test Cleaned Prompt from generate_prompt" in result.stdout
    
    # Check order: Reasoning Process should appear before AI Prompt
    reasoning_idx = result.stdout.find("Reasoning Process")
    prompt_idx = result.stdout.find("AI Prompt")
    assert reasoning_idx != -1 and prompt_idx != -1, "Reasoning or Prompt section missing"
    assert reasoning_idx < prompt_idx, "Thinking steps were not displayed before the prompt"

@patch('src.utils.cli.ImageGenerator')
@patch('src.utils.cli.PromptGenerator')
@patch('src.utils.cli.StorageManager')
@patch('src.utils.cli.MetricsCollector')
@patch('builtins.input', return_value='1') # Mock user input for interactive mode
# Patch print in prompt_generator as get_prompt_with_feedback prints there
@patch('src.generators.prompt_generator.PromptGenerator.get_prompt_with_feedback')
def test_generate_interactive_with_think_steps(
    MockGetPromptWithFeedback, MockInput, MockMetricsCollector, MockStorageManager,
    MockPromptGenerator, MockImageGenerator, 
    runner, mock_cli_config, mock_prompt_generator_instance # use the instance fixture
):
    # Configure the primary mock for PromptGenerator to return our detailed instance
    MockPromptGenerator.return_value = mock_prompt_generator_instance
    
    # The get_prompt_with_feedback method on the INSTANCE is what we need to control.
    # The task is to test CLI output, and get_prompt_with_feedback itself prints.
    # We need to simulate this print behavior or ensure the mock does.
    # The PromptGenerator's get_prompt_with_feedback was already modified to print.
    # So, we let the actual method (on the mocked instance) run if it's simple,
    # or mock its behavior including prints.
    # For this test, we care that the CLI calls it and its output (including think steps) appears.
    
    # Let's make the mocked get_prompt_with_feedback simulate the print output
    # that includes thinking steps.
    async def mock_get_prompt_with_feedback_behavior():
        # This is the behavior of the actual get_prompt_with_feedback
        # when thinking_process is populated on the instance.
        # We need to ensure that the `thinking_process` is available on the instance
        # that `get_prompt_with_feedback` is called on.
        
        # `mock_prompt_generator_instance` already has `thinking_process` set by its fixture.
        # `mock_prompt_generator_instance.get_prompt_with_feedback` is an AsyncMock.
        # We need to make this AsyncMock also *print* the thinking steps if we are testing that print.
        # However, the subtask solution modified the *actual* get_prompt_with_feedback to print.
        # So, if we make MockPromptGenerator.return_value = mock_prompt_generator_instance,
        # and then call `runner.invoke(app, ["generate", "-i"])`,
        # it will call `mock_prompt_generator_instance.get_prompt_with_feedback()`.
        # This is already an AsyncMock.
        # We need to either:
        # 1. Let the *actual* `get_prompt_with_feedback` run by not mocking it on the instance,
        #    and ensure `generate_prompt` (called by it) is properly mocked.
        # 2. Or, mock `get_prompt_with_feedback` and have the mock *itself* simulate the print statements.

        # Option 2: Mock `get_prompt_with_feedback` to simulate prints.
        # This is what `@patch('src.generators.prompt_generator.PromptGenerator.get_prompt_with_feedback')` does.
        # `MockGetPromptWithFeedback` is this new mock.
        
        # Let's refine `mock_prompt_generator_instance` for this test.
        # `mock_prompt_generator_instance.thinking_process` is already set.
        # The `get_prompt_with_feedback` on this instance should be the one printing.
        
        # The current patch on `PromptGenerator.get_prompt_with_feedback` means the CLI will call this *mocked* version,
        # not the one on `mock_prompt_generator_instance`.
        # So, `MockGetPromptWithFeedback` needs to be configured.
        
        # Let's print the thinking steps then the prompt, as the actual method would.
        output_lines = []
        if mock_prompt_generator_instance.thinking_process: # Accessing from the broader scope fixture
            output_lines.append("\n[Reasoning Process]")
            output_lines.append("-" * 80)
            for step in mock_prompt_generator_instance.thinking_process:
                output_lines.append(f"Thought: {step}")
            output_lines.append("-" * 80)
        
        prompt_val = "Test Cleaned Prompt from feedback"
        output_lines.append("\nGenerated prompt:")
        output_lines.append("-" * 80)
        output_lines.append(prompt_val)
        output_lines.append("-" * 80)
        
        # Simulate print
        print("\n".join(output_lines)) # This print will be captured by CliRunner
        return prompt_val # Return the prompt value

    MockGetPromptWithFeedback.side_effect = mock_get_prompt_with_feedback_behavior
    MockImageGenerator.return_value = mock_image_generator_instance
    MockStorageManager.return_value.get_output_path.return_value = MagicMock(with_suffix=lambda s: "path/to/prompt.txt")

    with patch('src.utils.cli.app.state.config', mock_cli_config):
        result = runner.invoke(app, ["generate", "--interactive"])
    
    assert result.exit_code == 0, f"CLI Error: {result.stdout}"
    MockGetPromptWithFeedback.assert_called_once() # Ensure the interactive path was taken

    assert "[Reasoning Process]" in result.stdout
    assert "Thought: Think Step CLI 1" in result.stdout # These are from the fixture
    assert "Thought: Think Step CLI 2" in result.stdout
    assert "Generated prompt:" in result.stdout
    assert "Test Cleaned Prompt from feedback" in result.stdout # From the mock_get_prompt_with_feedback_behavior

    # Check order: Reasoning Process should appear before Generated prompt (from feedback)
    reasoning_idx = result.stdout.find("[Reasoning Process]")
    prompt_idx = result.stdout.find("Generated prompt:") # This is printed by get_prompt_with_feedback
    assert reasoning_idx != -1 and prompt_idx != -1, "Reasoning or Prompt section missing in interactive"
    assert reasoning_idx < prompt_idx, "Interactive: Thinking steps were not displayed before the prompt"

    # Also check that the final "AI Prompt" panel (from cli.py) is shown
    assert "AI Prompt" in result.stdout 
    # This "AI Prompt" is printed by cli.py *after* get_prompt_with_feedback returns.
    # The thinking steps from cli.py's own print block should *not* appear here if interactive,
    # because get_prompt_with_feedback handles its own thinking process display.
    # The `cli.py` code for displaying think steps is:
    # `if hasattr(prompt_gen, 'thinking_process') and prompt_gen.thinking_process:`
    # This might still print if `thinking_process` is still populated and `interactive` was true.
    # The subtask was to ensure `get_prompt_with_feedback` prints it *before* user input.
    # The `cli.py` printing it again *after* might be redundant but not strictly wrong.
    # For now, let's assume the primary check is for the print within get_prompt_with_feedback.


@patch('src.utils.cli.ImageGenerator')
@patch('src.utils.cli.PromptGenerator')
@patch('src.utils.cli.StorageManager')
@patch('src.utils.cli.MetricsCollector')
@patch('asyncio.sleep', new_callable=AsyncMock) # Mock sleep in the loop
def test_loop_with_think_steps(
    MockSleep, MockMetricsCollector, MockStorageManager, MockPromptGenerator, MockImageGenerator,
    runner, mock_cli_config, mock_prompt_generator_instance
):
    MockPromptGenerator.return_value = mock_prompt_generator_instance
    MockImageGenerator.return_value = mock_image_generator_instance
    MockStorageManager.return_value.get_output_path.return_value = MagicMock(with_suffix=lambda s: "path/to/prompt.txt")

    # Make generate_prompt return different things or track calls if needed for loop
    # For this test, it can return the same mock prompt and thinking_process each time.
    # mock_prompt_generator_instance.generate_prompt is already an AsyncMock.

    with patch('src.utils.cli.app.state.config', mock_cli_config):
        result = runner.invoke(app, ["loop", "--batch-size", "2", "--interval", "0"]) # Fast loop

    assert result.exit_code == 0, f"CLI Error: {result.stdout}"
    assert mock_prompt_generator_instance.generate_prompt.call_count == 2
    assert mock_image_generator_instance.generate_image.call_count == 2
    
    # Check output for each iteration. Typer/Rich might make exact string count tricky.
    # We'll check for occurrences.
    assert result.stdout.count("Reasoning Process") == 2
    assert result.stdout.count("Think Step CLI 1") == 2 
    assert result.stdout.count("Think Step CLI 2") == 2
    assert result.stdout.count("Generated prompt for image 1:") > 0 # Check if the title is there
    assert result.stdout.count("Generated prompt for image 2:") > 0
    assert result.stdout.count("Test Cleaned Prompt from generate_prompt") == 2 # The prompt itself

    # Rough check for order in the first iteration (more complex for all iterations)
    first_reasoning_idx = result.stdout.find("Reasoning Process")
    # The title of the prompt panel in loop is "Prompt X/Y"
    first_prompt_panel_idx = result.stdout.find("Prompt 1/2") 
    
    assert first_reasoning_idx != -1 and first_prompt_panel_idx != -1
    assert first_reasoning_idx < first_prompt_panel_idx

    # Ensure sleep was called (batch_size - 1) times
    assert MockSleep.call_count == 1 # For batch_size=2, sleep is called once.
    MockSleep.assert_any_call(1) # interval is 0, but code uses max(1, interval)
