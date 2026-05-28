pipeline {
    agent any

    options {
        disableConcurrentBuilds()
        timestamps()
    }

    parameters {
        string(name: 'GIT_BRANCH', defaultValue: 'main', description: 'Git branch to build and deploy.')
        string(name: 'GITHUB_CREDENTIALS_ID', defaultValue: 'github-pat-token', description: 'Jenkins username/password GitHub PAT credential used for checkout and GHCR login.')
        string(name: 'DEPLOY_HOST', defaultValue: '100.102.25.32', description: 'SSH host for the Jetson Orin Nano server.')
        string(name: 'DEPLOY_PATH', defaultValue: '/opt/finbert', description: 'Deployment directory on the remote host.')
        string(name: 'IMAGE_NAME', defaultValue: 'ghcr.io/holofoundry/finbert-sentiment-analyser', description: 'GHCR image name to build and deploy.')
        string(name: 'TARGET_PLATFORM', defaultValue: 'linux/arm64', description: 'Docker platform to build.')
        string(name: 'SSH_CREDENTIALS_ID', defaultValue: '1d7fca85-f028-47e9-8fb2-b4e81978c67a', description: 'Jenkins SSH private key credential used to deploy to the webserver.')
    }

    environment {
        REPOSITORY_URL = 'https://github.com/holofoundry/finbert-sentiment-analyser.git'
    }

    stages {
        stage('Checkout') {
            steps {
                script {
                    env.DEPLOY_GITHUB_CREDENTIALS_ID = params.GITHUB_CREDENTIALS_ID?.trim() ?: 'github-pat-token'
                }
                deleteDir()
                git branch: "${params.GIT_BRANCH}",
                    credentialsId: "${env.DEPLOY_GITHUB_CREDENTIALS_ID}",
                    url: "${env.REPOSITORY_URL}"
            }
        }

        stage('Prepare') {
            steps {
                script {
                    env.DEPLOY_WEB_HOST = params.DEPLOY_HOST?.trim() ?: '100.102.25.32'
                    env.DEPLOY_REMOTE_APP_DIR = params.DEPLOY_PATH?.trim() ?: '/opt/finbert'
                    env.DEPLOY_IMAGE_NAME = params.IMAGE_NAME?.trim() ?: 'ghcr.io/holofoundry/finbert-sentiment-analyser'
                    env.DEPLOY_TARGET_PLATFORM = params.TARGET_PLATFORM?.trim() ?: 'linux/arm64'
                    env.DEPLOY_IMAGE_TAG = 'latest'
                    env.DEPLOY_SSH_CREDENTIALS_ID = params.SSH_CREDENTIALS_ID?.trim() ?: '1d7fca85-f028-47e9-8fb2-b4e81978c67a'

                    writeFile file: '.deploy.env', text: """FINBERT_IMAGE=${env.DEPLOY_IMAGE_NAME}
FINBERT_IMAGE_TAG=${env.DEPLOY_IMAGE_TAG}
FINBERT_HTTP_PORT=8081
"""
                }
            }
        }

        stage('Build and Push Image') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: "${env.DEPLOY_GITHUB_CREDENTIALS_ID}",
                    usernameVariable: 'GHCR_USER',
                    passwordVariable: 'GHCR_TOKEN'
                )]) {
                    sh '''
                        set -eu
                        printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin

                        if docker buildx version >/dev/null 2>&1; then
                            docker buildx create --use --name finbert-builder 2>/dev/null || docker buildx use finbert-builder
                            docker buildx build \
                                --platform "$DEPLOY_TARGET_PLATFORM" \
                                --file Dockerfile \
                                --tag "$DEPLOY_IMAGE_NAME:latest" \
                                --push \
                                .
                        else
                            docker build --pull \
                                --platform "$DEPLOY_TARGET_PLATFORM" \
                                --file Dockerfile \
                                --tag "$DEPLOY_IMAGE_NAME:latest" \
                                .
                            docker push "$DEPLOY_IMAGE_NAME:latest"
                        fi
                    '''
                }
            }
        }

        stage('Deploy') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: "${env.DEPLOY_GITHUB_CREDENTIALS_ID}",
                    usernameVariable: 'GHCR_USER',
                    passwordVariable: 'GHCR_TOKEN'
                ), sshUserPrivateKey(
                    credentialsId: "${env.DEPLOY_SSH_CREDENTIALS_ID}",
                    keyFileVariable: 'SSH_KEY',
                    usernameVariable: 'SSH_USER'
                )]) {
                    sh '''
                        set -eu

                        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
                            "$SSH_USER@$DEPLOY_WEB_HOST" \
                            "mkdir -p '$DEPLOY_REMOTE_APP_DIR'"

                        scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
                            docker-compose.pi.yml \
                            "$SSH_USER@$DEPLOY_WEB_HOST:$DEPLOY_REMOTE_APP_DIR/docker-compose.yml"

                        scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
                            .deploy.env \
                            "$SSH_USER@$DEPLOY_WEB_HOST:$DEPLOY_REMOTE_APP_DIR/.deploy.env"

                        printf '%s' "$GHCR_TOKEN" | ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
                            "$SSH_USER@$DEPLOY_WEB_HOST" \
                            "docker login ghcr.io -u '$GHCR_USER' --password-stdin"

                        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
                            "$SSH_USER@$DEPLOY_WEB_HOST" \
                            "set -eu
                             cd '$DEPLOY_REMOTE_APP_DIR'
                             touch .env
                             docker compose --env-file .env --env-file .deploy.env pull
                             docker compose --env-file .env --env-file .deploy.env up -d --remove-orphans
                             docker image prune -f"
                    '''
                }
            }
        }
    }
}
