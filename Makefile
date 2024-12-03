.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y wattsquad || :
	@pip install -e .

run_preprocess:
	python -c 'from taxifare.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from taxifare.interface.main import train; train()'

run_pred:
	python -c 'from taxifare.interface.main import pred; pred()'

run_evaluate:
	python -c 'from taxifare.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m taxifare.interface.workflow

run_api:
	uvicorn taxifare.api.fast:app --reload

##################### TESTS #####################
test_gcp_setup:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_project \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_wagon_project


test_api_root:
	pytest \
	tests/api/test_endpoints.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_root_returns_greeting --asyncio-mode=strict -W "ignore"

test_api_predict:
	pytest \
	tests/api/test_endpoints.py::test_predict_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_is_dict --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_has_key --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_val_is_float --asyncio-mode=strict -W "ignore"

test_api_on_docker:
	pytest \
	tests/api/test_docker_endpoints.py --asyncio-mode=strict -W "ignore"

test_api_on_prod:
	pytest \
	tests/api/test_cloud_endpoints.py --asyncio-mode=strict -W "ignore"





reset_all_files: reset_local_files reset_bq_files reset_gcs_files


##################### CLEANING #####################

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints


# DOCKER DEPLOYMENT

build_container_local:
	docker build --tag=$$IMAGE:dev .

test_run:
	docker run --env-file .env -p 8000:8000 $$IMAGE:dev

build_for_production:
	docker build -t  $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$DOCKER_REPO_NAME/$$IMAGE:prod .

## Step 1
allow_docker_push:
	gcloud auth configure-docker $$GCP_REGION-docker.pkg.dev

# Step 3
build_for_production:
	docker build -t  $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$DOCKER_REPO_NAME/$$IMAGE:prod .

## Step 4
push_image_production:
	docker push $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$DOCKER_REPO_NAME/$$IMAGE:prod

# Step 5
deploy_to_cloud_run:
	gcloud run deploy --image $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$DOCKER_REPO_NAME/$$IMAGE:prod --memory $$MEMORY --region $$GCP_REGION

## STREAMLIT

default: pytest

# default: pylint pytest

# pylint:
# 	find . -iname "*.py" -not -path "./tests/test_*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	echo "no tests"

# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

streamlit:
	-@streamlit run wattsquad/frontend/app.py


# ----------------------------------
#    LOCAL INSTALL COMMANDS
# ----------------------------------
install:
	@pip install . -U

clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr *.dist-info
	@rm -fr *.egg-info
	-@rm model.joblib
