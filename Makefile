
ANDROID_SDK_ROOT ?= $(HOME)/Library/Android/sdk
EMULATOR = $(ANDROID_SDK_ROOT)/emulator/emulator
AVD_NAME = pixel_7_api_35

BUILD_DIR = build
INSTALL_DIR = $(HOME)/.local/bin

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) *.cpython-310-darwin.so GUI

.PHONY: build
build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ..
	$(MAKE) -C $(BUILD_DIR)

.PHONY: install
install:
	# pyenv install 3.12.7 || true
	# pyenv global 3.12.7
	# python -m ensurepip --upgrade
	# python -m pip install --upgrade pip setuptools wheel

	# LDFLAGS="-L/opt/homebrew/opt/llvm/lib" \
    # CPPFLAGS="-I/opt/homebrew/opt/llvm/include" \
	# pip install --use-pep517 --no-binary symengine symengine==0.9.2

	pip install -r requirements.txt --use-pep517
ifeq ($(shell uname), Darwin)
		xcode-select --install || (sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer && \
		sudo xcodebuild -runFirstLaunch)
	xcrun simctl create "iOS17.2" "iPhone 15" "iOS17.2" 2>/dev/null || true

	# rm -rf GUI/obj GUI/bin || true
	# sudo dotnet workload update

	brew update
	brew install mpfr libmpc gh git libomp yaml-cpp gmp gsl cmake boost android-commandlinetools openjdk@17 git-lfs

	sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk

	rm -rf $$HOME/Library/Android/sdk/cmdline-tools/latest || true
	rm -rf $$HOME/Library/Android/sdk/cmdline-tools/temp || true
	mkdir -p $$HOME/Library/Android/sdk/cmdline-tools
	curl -o commandlinetools.zip https://dl.google.com/android/repository/commandlinetools-mac-9477386_latest.zip
	unzip -o commandlinetools.zip -d $$HOME/Library/Android/sdk/cmdline-tools/temp ||true
	mv $$HOME/Library/Android/sdk/cmdline-tools/temp/cmdline-tools $$HOME/Library/Android/sdk/cmdline-tools/latest || true
	rm -rf $$HOME/Library/Android/sdk/cmdline-tools/temp
	rm -f commandlinetools.zip

	sdkmanager "platforms;android-35" "build-tools;35.0.0" "platform-tools" "emulator" "system-images;android-35;google_apis;arm64-v8a"
	sdkmanager --licenses
	export ANDROID_HOME=$$HOME/Library/Android/sdk
	export PATH=$$PATH:$$ANDROID_HOME/tools
	export PATH=$$PATH:$$ANDROID_HOME/platform-tools
	export JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home
	echo "export JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home" >> ~/.zshrc
	echo 'export PATH="$$PATH:/usr/local/share/dotnet"' >> ~/.zshrc
	source ~/.zshrc

	yes | $$HOME/Library/Android/sdk/cmdline-tools/latest/bin/sdkmanager --update
	yes | $$HOME/Library/Android/sdk/cmdline-tools/latest/bin/sdkmanager "platforms;android-35" "build-tools;35.0.0" "platform-tools" "cmdline-tools;12.0" "emulator" "system-images;android-35;google_apis;arm64-v8a"
	yes | $$HOME/Library/Android/sdk/cmdline-tools/latest/bin/sdkmanager --licenses

	# sudo dotnet workload install maui
	# sudo dotnet new install Microsoft.Maui.Templates
	# sudo dotnet workload list
	# sudo dotnet workload repair
	# rm -rf GUI || true
	# dotnet new maui -n GUI --force
	# sed -i '' 's/<UseMaui>true<\/UseMaui>/<UseMaui>true<\/UseMaui>\n    <LinkMode>None<\/LinkMode>/' GUI/GUI.csproj
	# sudo dotnet build GUI/GUI.csproj /p:JavaSdkDirectory="/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home"
	# sudo dotnet build GUI/GUI.csproj -f net9.0-android /p:JavaSdkDirectory="/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home"
	# # sudo dotnet build -t:CheckDotNetMAUIWorkloads

else ifeq ($(shell uname), Linux)
	if [ -f /etc/fedora-release ]; then \
		sudo dnf install -y libomp yaml-cpp gmp-devel gsl-devel cmake boost-devel; \
	else \
		sudo snap install cmake --classic; \
		export PATH=$$PATH:/snap/bin; \
		sudo apt-get update; \
		sudo apt-get install -y libomp-dev libyaml-cpp-dev libgmp-dev libgsl-dev cmake libboost-all-dev; \
	fi
endif

.PHONY: start-emulator
start-emulator:
	$(EMULATOR) \
		-avd $(AVD_NAME) \
		-no-audio \
		-no-window \
		-gpu swiftshader_indirect \
		-memory 2048 \
		-no-snapshot \
		-netfast &

.PHONY: kill-emulator
kill-emulator:
	@adb devices | grep emulator | cut -f1 | xargs -I {} adb -s {} emu kill

.PHONY: wait-emulator
wait-emulator:
	@adb wait-for-device shell "while [ -z \$$(getprop sys.boot_completed) ]; do sleep 1; done"

.PHONY: check-emulator
check-emulator:
	@adb devices
	@adb shell getprop sys.boot_completed

.PHONY: delete-avd
delete-avd:
	@avdmanager delete avd --name "$(AVD_NAME)"

.PHONY: create-avd
create-avd:
	avdmanager --verbose create avd \
		--name "$(AVD_NAME)" \
		--package "system-images;android-35;google_apis;arm64-v8a" \
		--device "pixel_7" \
		--force \
		--sdcard 512M

.PHONY: setup-maui
setup-maui:
	sudo dotnet workload install maui
	sudo dotnet workload install android
	sudo dotnet workload install maui-android

.PHONY: list-avds
list-avds:
	@avdmanager list avd

.PHONY: list-devices
list-devices:
	@avdmanager list device

.PHONY: check-android-env
check-android-env: setup-maui
	@echo "Checking Android environment..."
	@command -v dotnet >/dev/null 2>&1 || { echo "dotnet is not installed"; exit 1; }
	@dotnet --list-sdks
	@echo "Android SDK Location: $(ANDROID_HOME)"
	@echo "Java Home: $(JAVA_HOME)"
	@$(ANDROID_HOME)/cmdline-tools/latest/bin/sdkmanager --list
	@$(ANDROID_HOME)/cmdline-tools/latest/bin/avdmanager list avd


.PHONY: setup
setup:
	chmod +x ./bin/bash/convert_to_csv.zsh
	export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

.PHONY: run
run:
	python main.py

.PHONY: help
help:
	@echo "利用可能なターゲット:"
	@echo "  build         - プロジェクトをビルド"
	@echo "  clean         - ビルド成果物を削除"
	@echo "  install       - システムにインストール"
	@echo "  start-emulator- Androidエミュレータを起動"
	@echo "  kill-emulator - エミュレータを終了"