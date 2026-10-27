// Copyright The MatrixHub Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package hf

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/matrixhub-ai/hfd/pkg/authenticate"
	"github.com/matrixhub-ai/hfd/pkg/repository"
	"github.com/matrixhub-ai/hfd/pkg/storage"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"

	domaingit "github.com/matrixhub-ai/matrixhub/internal/domain/git"
	gitmocks "github.com/matrixhub-ai/matrixhub/internal/domain/git/mocks"
	modelmocks "github.com/matrixhub-ai/matrixhub/internal/domain/model/mocks"
	"github.com/matrixhub-ai/matrixhub/internal/domain/robot"
	"github.com/matrixhub-ai/matrixhub/internal/infra/authcodec"
)

func TestHandleCommitRoutesByRepositoryType(t *testing.T) {
	repoTypes := []struct {
		repoType    string
		storageName string
	}{
		{repoType: "models", storageName: "test-user/shared-name"},
		{repoType: "datasets", storageName: "datasets/test-user/shared-name"},
		{repoType: "spaces", storageName: "spaces/test-user/shared-name"},
	}
	outcomes := []struct {
		name       string
		commitHash string
		err        error
		wantStatus int
	}{
		{name: "success", commitHash: "commit-sha", wantStatus: http.StatusOK},
		{name: "commit error", err: errors.New("commit failed"), wantStatus: http.StatusInternalServerError},
	}

	for _, rt := range repoTypes {
		for _, outcome := range outcomes {
			t.Run(rt.repoType+"/"+outcome.name, func(t *testing.T) {
				ctrl := gomock.NewController(t)
				modelService := modelmocks.NewMockIModelService(ctrl)
				gitRepo := gitmocks.NewMockIGitRepo(ctrl)
				commit := &domaingit.Commit{
					Message:     "Add readme\n\nTest description",
					AuthorName:  "HuggingFace",
					AuthorEmail: "hf@users.noreply.huggingface.co",
				}
				ops := []domaingit.CommitOperation{{
					Type:    domaingit.CommitOperationAdd,
					Path:    "README.md",
					Content: []byte("readme content\n"),
				}}
				if rt.repoType == "models" {
					modelService.EXPECT().
						CreateModelCommit(gomock.Any(), "test-user", "shared-name", "main", commit, ops).
						Return(outcome.commitHash, outcome.err)
				} else {
					gitRepo.EXPECT().
						CreateCommit(gomock.Any(), rt.repoType, "test-user", "shared-name", "main", commit, ops).
						Return(outcome.commitHash, outcome.err)
				}

				testStorage := storage.NewStorage(storage.WithRootDir(t.TempDir()))
				_, err := repository.Init(t.Context(), testStorage.ResolvePath(rt.storageName), "main")
				require.NoError(t, err)
				handler := NewHandler(
					WithStorage(testStorage),
					WithServices(modelService, gitRepo, nil),
				)
				body := "{\"key\":\"header\",\"value\":{\"summary\":\"Add readme\",\"description\":\"Test description\"}}\n" +
					"{\"key\":\"file\",\"value\":{\"content\":\"readme content\\n\",\"path\":\"README.md\",\"encoding\":\"utf-8\"}}\n"
				request := httptest.NewRequest(http.MethodPost,
					"http://example.com/api/"+rt.repoType+"/test-user/shared-name/commit/main", strings.NewReader(body))
				request.Header.Set("Content-Type", "application/x-ndjson")
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, request)

				require.Equal(t, outcome.wantStatus, response.Code, response.Body.String())
				if outcome.err != nil {
					require.Contains(t, response.Body.String(), outcome.err.Error())
					return
				}
				var result commitResponse
				require.NoError(t, json.NewDecoder(response.Body).Decode(&result))
				require.Equal(t, commitResponse{
					CommitURL:     "http://example.com/" + rt.storageName + "/commit/" + outcome.commitHash,
					CommitOid:     outcome.commitHash,
					CommitMessage: commit.Message,
				}, result)
			})
		}
	}
}

func TestCommitAuthorName(t *testing.T) {
	robotIdentity := robot.NewRobotIdentity(4, "robot$test-pro-select")
	encodedRobot, err := authcodec.Marshal(robotIdentity)
	if err != nil {
		t.Fatalf("marshal robot identity: %v", err)
	}

	tests := []struct {
		name string
		user authenticate.UserInfo
		want string
	}{
		{
			name: "typed identity",
			user: authenticate.UserInfo{User: encodedRobot},
			want: "robot$test-pro-select",
		},
		{
			name: "plain user name",
			user: authenticate.UserInfo{User: "HuggingFace"},
			want: "HuggingFace",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := commitAuthorName(tt.user); got != tt.want {
				t.Fatalf("commitAuthorName() = %q, want %q", got, tt.want)
			}
		})
	}
}
