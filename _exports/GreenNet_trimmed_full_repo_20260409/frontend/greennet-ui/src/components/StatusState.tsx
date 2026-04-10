type StateNoticeProps = {
  title: string;
  description?: string;
};

export function LoadingNotice({ title, description }: StateNoticeProps) {
  return (
    <div className="glass-card state-notice" role="status" aria-live="polite">
      <div className="spinner" aria-hidden />
      <div>
        <strong>{title}</strong>
        {description ? <p>{description}</p> : null}
      </div>
    </div>
  );
}

export function ErrorNotice({ title, description }: StateNoticeProps) {
  return (
    <div className="glass-card state-notice error" role="alert">
      <div>
        <strong>{title}</strong>
        {description ? <p>{description}</p> : null}
      </div>
    </div>
  );
}

export function InfoNotice({ title, description }: StateNoticeProps) {
  return (
    <div className="glass-card state-notice info" role="status" aria-live="polite">
      <div>
        <strong>{title}</strong>
        {description ? <p>{description}</p> : null}
      </div>
    </div>
  );
}
