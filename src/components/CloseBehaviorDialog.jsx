// Close behavior dialog component for first-time close action selection.

import React from "react";

export function CloseBehaviorDialog({ onChoice }) {
  return (
    <div className="modal-backdrop" role="presentation">
      <section
        className="close-behavior-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="close-behavior-title"
      >
        <header className="close-behavior-header">
          <h2 id="close-behavior-title">关闭行为设置</h2>
          <p>点击关闭按钮时，您希望程序如何处理？</p>
        </header>
        <div className="close-behavior-options">
          <button
            className="close-behavior-option primary"
            type="button"
            onClick={() => onChoice(true)}
          >
            <div className="close-behavior-option-icon">🖥️</div>
            <div className="close-behavior-option-content">
              <strong>退到后台（推荐）</strong>
              <span>程序将在系统托盘中保留，可以快速恢复</span>
            </div>
          </button>
          <button
            className="close-behavior-option secondary"
            type="button"
            onClick={() => onChoice(false)}
          >
            <div className="close-behavior-option-icon">❌</div>
            <div className="close-behavior-option-content">
              <strong>直接退出</strong>
              <span>程序将完全退出</span>
            </div>
          </button>
        </div>
      </section>
    </div>
  );
}
